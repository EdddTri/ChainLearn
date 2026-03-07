// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title FLCoordinator
 * @notice Coordinates federated ensemble learning across hospital nodes.
 * @dev
 *   Proof-of-Capacity (PoC) benchmark fingerprints are verified on registration
 *   to assign each hospital a capacity class (Weak / Medium / Strong).
 *   That class deterministically maps to one of three model types:
 *     Weak   -> Light   (e.g. MobileNet)
 *     Medium -> Medium  (e.g. EfficientNet-B0)
 *     Strong -> Heavy   (e.g. ResNet-50)
 *
 *   After local training each hospital submits a hash of its model weights
 *   together with reliability metrics (confidence, ECE) and the model type it
 *   trained.  The contract validates that the submitted model type matches the
 *   hospital's assigned capacity class.
 *
 *   Weights for ensemble aggregation are computed on-chain using capacity,
 *   reliability, and participation history.  The final ensemble prediction
 *   hash is recorded on-chain for auditability.
 *
 *   All percentages use fixed-point integer math with SCALE = 10 000
 *   (i.e. 10 000 = 100%).
 */
contract FLCoordinator {
    // -- Constants ------------------------------------------------------------
    uint256 public constant SCALE = 10_000;        // 100 % in fixed-point
    uint256 public constant MAX_WEIGHT = 15_000;   // 1.5x cap

    uint256 public constant CAPACITY_WEAK   =  8_000;  // 0.8x
    uint256 public constant CAPACITY_MEDIUM = 10_000;  // 1.0x
    uint256 public constant CAPACITY_STRONG = 12_000;  // 1.2x

    // Participation bonus: +500 per round, capped at 2500 (+25% max)
    uint256 public constant PARTICIPATION_BONUS_PER_ROUND = 500;
    uint256 public constant MAX_PARTICIPATION_BONUS = 2_500;

    // -- Types ----------------------------------------------------------------
    enum CapacityClass { Weak, Medium, Strong }

    /// @notice Model architectures assigned by capacity class.
    enum ModelType { Light, Medium, Heavy }

    struct Hospital {
        address       addr;
        string        name;
        CapacityClass capacityClass;
        bool          isRegistered;
        // PoC benchmark fingerprint (SHA-256 of throughput + params)
        bytes32       pocBenchmarkHash;
        // Latest reliability metrics (updated every submission)
        uint256       confidence;  // [0, SCALE]
        uint256       ece;         // [0, SCALE]
        // Participation history
        uint256       roundsParticipated;
    }

    struct Submission {
        bytes32   modelHash;
        uint256   confidence;
        uint256   ece;
        uint256   timestamp;
        ModelType modelType;
    }

    struct EnsembleRecord {
        bytes32 predictionHash;
        uint256 participantCount;
        uint256 timestamp;
    }

    // -- State ----------------------------------------------------------------
    address public owner;
    uint256 public currentRound;

    mapping(address => Hospital) public hospitals;
    address[] public hospitalList;

    // round => hospital => submission
    mapping(uint256 => mapping(address => Submission)) public roundSubmissions;
    // round => list of submitters (for enumeration)
    mapping(uint256 => address[]) public roundSubmitters;
    // round => ensemble record
    mapping(uint256 => EnsembleRecord) public ensembleRecords;

    // -- Events ---------------------------------------------------------------
    event HospitalRegistered(
        address indexed hospital,
        string name,
        CapacityClass capacityClass,
        bytes32 pocBenchmarkHash
    );
    event RoundStarted(uint256 indexed round);
    event UpdateSubmitted(
        uint256   indexed round,
        address   indexed hospital,
        bytes32   modelHash,
        uint256   confidence,
        uint256   ece,
        ModelType modelType
    );
    event EnsemblePredictionRecorded(
        uint256 indexed round,
        bytes32 predictionHash,
        uint256 participantCount
    );

    // -- Modifiers ------------------------------------------------------------
    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    modifier onlyRegistered() {
        require(hospitals[msg.sender].isRegistered, "Not a registered hospital");
        _;
    }

    // -- Constructor ----------------------------------------------------------
    constructor() {
        owner = msg.sender;
        currentRound = 0;
    }

    // -- Signature helpers ---------------------------------------------------
    function _toEthSignedMessageHash(
        bytes32 _hash
    ) internal pure returns (bytes32) {
        return keccak256(
            abi.encodePacked("\x19Ethereum Signed Message:\n32", _hash)
        );
    }

    function _recoverSigner(
        bytes32 _ethSignedHash,
        bytes memory _signature
    ) internal pure returns (address) {
        require(_signature.length == 65, "Invalid signature length");

        bytes32 r;
        bytes32 s;
        uint8 v;

        assembly {
            r := mload(add(_signature, 32))
            s := mload(add(_signature, 64))
            v := byte(0, mload(add(_signature, 96)))
        }

        if (v < 27) {
            v += 27;
        }
        require(v == 27 || v == 28, "Invalid signature v");

        return ecrecover(_ethSignedHash, v, r, s);
    }

    // -- Model-type assignment ------------------------------------------------
    /**
     * @notice Return the model type assigned to a given capacity class.
     *         Weak -> Light, Medium -> Medium, Strong -> Heavy.
     */
    function getModelType(
        CapacityClass _class
    ) public pure returns (ModelType) {
        if (_class == CapacityClass.Weak)   return ModelType.Light;
        if (_class == CapacityClass.Medium) return ModelType.Medium;
        return ModelType.Heavy;
    }

    // -- Hospital Registration ------------------------------------------------
    /**
     * @notice Register a hospital node with a capacity class and PoC benchmark hash.
     * @param _hospital       Address of the hospital node.
     * @param _name           Human-readable hospital identifier.
     * @param _capacity       Capacity tier: Weak (0), Medium (1), Strong (2).
     * @param _benchmarkHash  SHA-256 hash of the PoC benchmark results
     *                        (throughput, k_steps, batch_size, capacity_class),
     *                        signed by the hospital's node identity.
     * @param _benchmarkSignature  ECDSA signature over _benchmarkHash produced
     *                             by _hospital's private key (EIP-191 format).
     */
    function registerHospital(
        address _hospital,
        string calldata _name,
        CapacityClass _capacity,
        bytes32 _benchmarkHash,
        bytes calldata _benchmarkSignature
    ) external onlyOwner {
        require(!hospitals[_hospital].isRegistered, "Hospital already registered");
        address recovered = _recoverSigner(
            _toEthSignedMessageHash(_benchmarkHash),
            _benchmarkSignature
        );
        require(recovered == _hospital, "Invalid benchmark signature");

        hospitals[_hospital] = Hospital({
            addr: _hospital,
            name: _name,
            capacityClass: _capacity,
            isRegistered: true,
            pocBenchmarkHash: _benchmarkHash,
            confidence: 0,
            ece: 0,
            roundsParticipated: 0
        });
        hospitalList.push(_hospital);

        emit HospitalRegistered(_hospital, _name, _capacity, _benchmarkHash);
    }

    // -- Round Lifecycle ------------------------------------------------------
    /**
     * @notice Advance to a new training round.
     */
    function startNewRound() external onlyOwner {
        require(hospitalList.length > 0, "No hospitals registered");
        currentRound++;
        emit RoundStarted(currentRound);
    }

    // -- Model Update Submission ----------------------------------------------
    /**
     * @notice Submit a trained model hash along with self-reported metrics.
     * @param _modelHash   Cryptographic hash (SHA-256 / IPFS CID) of model weights.
     * @param _confidence  Self-reported confidence in [0, SCALE].
     * @param _ece         Self-reported Expected Calibration Error in [0, SCALE].
     * @param _modelType   Model architecture used -- must match the hospital's
     *                     capacity-assigned type (Light / Medium / Heavy).
     */
    function submitUpdate(
        bytes32 _modelHash,
        uint256 _confidence,
        uint256 _ece,
        ModelType _modelType
    ) external onlyRegistered {
        require(currentRound > 0, "No active round");
        require(_confidence <= SCALE, "Confidence out of range");
        require(_ece <= SCALE, "ECE out of range");
        require(
            roundSubmissions[currentRound][msg.sender].timestamp == 0,
            "Already submitted this round"
        );

        // Enforce that the hospital trained the model assigned to its capacity
        ModelType assigned = getModelType(hospitals[msg.sender].capacityClass);
        require(_modelType == assigned, "Wrong model type for capacity class");

        // Store submission
        roundSubmissions[currentRound][msg.sender] = Submission({
            modelHash: _modelHash,
            confidence: _confidence,
            ece: _ece,
            timestamp: block.timestamp,
            modelType: _modelType
        });
        roundSubmitters[currentRound].push(msg.sender);

        // Update latest reliability metrics on the hospital record
        hospitals[msg.sender].confidence = _confidence;
        hospitals[msg.sender].ece = _ece;

        // Increment participation counter
        hospitals[msg.sender].roundsParticipated++;

        emit UpdateSubmitted(
            currentRound,
            msg.sender,
            _modelHash,
            _confidence,
            _ece,
            _modelType
        );
    }

    // -- Ensemble Prediction Recording ----------------------------------------
    /**
     * @notice Record the hash of the final ensemble prediction for a round.
     * @dev    Called by the aggregator after computing the weighted ensemble.
     * @param _predictionHash   SHA-256 / IPFS CID of the ensemble output.
     * @param _participantCount Number of hospitals included in the ensemble.
     */
    function recordEnsemblePrediction(
        bytes32 _predictionHash,
        uint256 _participantCount
    ) external onlyOwner {
        require(currentRound > 0, "No active round");
        require(
            ensembleRecords[currentRound].timestamp == 0,
            "Ensemble already recorded for this round"
        );
        require(_participantCount > 0, "No participants");

        ensembleRecords[currentRound] = EnsembleRecord({
            predictionHash: _predictionHash,
            participantCount: _participantCount,
            timestamp: block.timestamp
        });

        emit EnsemblePredictionRecorded(
            currentRound,
            _predictionHash,
            _participantCount
        );
    }

    // -- Weight Calculation ---------------------------------------------------
    /**
     * @notice Return the capacity multiplier for a given class.
     */
    function _capacityMultiplier(
        CapacityClass _class
    ) internal pure returns (uint256) {
        if (_class == CapacityClass.Weak)   return CAPACITY_WEAK;
        if (_class == CapacityClass.Medium) return CAPACITY_MEDIUM;
        return CAPACITY_STRONG;
    }

    /**
     * @notice Deterministic weight calculation with participation bonus.
     * @dev    baseWeight = capMul * confidence * (SCALE - ece) / SCALE^2
     *         bonus      = min(roundsParticipated * 500, 2500)
     *         w_i        = min(baseWeight + bonus, MAX_WEIGHT)
     *
     *         All inputs are fixed-point with SCALE = 10 000.
     *
     * @param _capacityClass       The hospital's tier.
     * @param _confidence          Confidence metric in [0, SCALE].
     * @param _ece                 Expected Calibration Error in [0, SCALE].
     * @param _roundsParticipated  Number of rounds this hospital has submitted in.
     * @return weight              The aggregation weight (fixed-point, SCALE = 10 000).
     */
    function calculateWeightPure(
        CapacityClass _capacityClass,
        uint256 _confidence,
        uint256 _ece,
        uint256 _roundsParticipated
    ) public pure returns (uint256 weight) {
        require(_confidence <= SCALE, "Confidence out of range");
        require(_ece <= SCALE, "ECE out of range");

        uint256 capMul = _capacityMultiplier(_capacityClass);

        // raw = capMul * confidence * (SCALE - ece)
        uint256 raw = capMul * _confidence * (SCALE - _ece);

        // Divide by SCALE^2 to normalise back to fixed-point SCALE
        uint256 baseWeight = raw / (SCALE * SCALE);

        // Participation bonus: +500 per round participated, max +2500
        uint256 bonus = _roundsParticipated * PARTICIPATION_BONUS_PER_ROUND;
        if (bonus > MAX_PARTICIPATION_BONUS) {
            bonus = MAX_PARTICIPATION_BONUS;
        }

        weight = baseWeight + bonus;

        // Apply cap
        if (weight > MAX_WEIGHT) {
            weight = MAX_WEIGHT;
        }
    }

    /**
     * @notice Calculate the aggregation weight for a registered hospital
     *         using its stored capacity class, reliability metrics, and
     *         participation history.
     * @param _hospital  Address of the registered hospital.
     * @return weight    The capped aggregation weight.
     */
    function calculateWeight(
        address _hospital
    ) external view returns (uint256 weight) {
        require(hospitals[_hospital].isRegistered, "Hospital not registered");

        Hospital storage h = hospitals[_hospital];
        weight = calculateWeightPure(
            h.capacityClass,
            h.confidence,
            h.ece,
            h.roundsParticipated
        );
    }

    // -- View Helpers ---------------------------------------------------------

    function getHospitalCount() external view returns (uint256) {
        return hospitalList.length;
    }

    function getHospitalInfo(
        address _hospital
    )
        external
        view
        returns (
            string memory name,
            CapacityClass capacityClass,
            ModelType     assignedModelType,
            uint256 confidence,
            uint256 ece,
            bool isRegistered,
            bytes32 pocBenchmarkHash,
            uint256 roundsParticipated
        )
    {
        Hospital storage h = hospitals[_hospital];
        return (
            h.name,
            h.capacityClass,
            getModelType(h.capacityClass),
            h.confidence,
            h.ece,
            h.isRegistered,
            h.pocBenchmarkHash,
            h.roundsParticipated
        );
    }

    function getRoundSubmission(
        uint256 _round,
        address _hospital
    )
        external
        view
        returns (
            bytes32   modelHash,
            uint256   confidence,
            uint256   ece,
            uint256   timestamp,
            ModelType modelType
        )
    {
        Submission storage s = roundSubmissions[_round][_hospital];
        return (s.modelHash, s.confidence, s.ece, s.timestamp, s.modelType);
    }

    function getRoundSubmitterCount(
        uint256 _round
    ) external view returns (uint256) {
        return roundSubmitters[_round].length;
    }

    function getEnsembleRecord(
        uint256 _round
    )
        external
        view
        returns (
            bytes32 predictionHash,
            uint256 participantCount,
            uint256 timestamp
        )
    {
        EnsembleRecord storage r = ensembleRecords[_round];
        return (r.predictionHash, r.participantCount, r.timestamp);
    }
}
