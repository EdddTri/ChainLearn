const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("FLCoordinator", function () {
    let contract;
    let owner, hospitalA, hospitalB, hospitalC, nonOwner;

    const SCALE = 10_000n;
    const MAX_WEIGHT = 15_000n;

    const CAP_WEAK = 8_000n;
    const CAP_MEDIUM = 10_000n;
    const CAP_STRONG = 12_000n;

    const BONUS_PER_ROUND = 500n;
    const MAX_BONUS = 2_500n;

    // CapacityClass enum values
    const Weak = 0;
    const Medium = 1;
    const Strong = 2;

    // ModelType enum values
    const Light = 0;
    const MediumModel = 1;
    const Heavy = 2;

    // Dummy PoC benchmark hash
    const benchmarkHash = ethers.keccak256(ethers.toUtf8Bytes("poc-benchmark-v1"));
    async function sigFor(signer, hash = benchmarkHash) {
        return await signer.signMessage(ethers.getBytes(hash));
    }

    /**
     * Mirror the on-chain weight formula in JS for test assertions.
     */
    function expectedWeight(capMul, confidence, ece, roundsParticipated = 0n) {
        const raw = capMul * confidence * (SCALE - ece);
        const baseWeight = raw / (SCALE * SCALE);
        let bonus = roundsParticipated * BONUS_PER_ROUND;
        if (bonus > MAX_BONUS) bonus = MAX_BONUS;
        let w = baseWeight + bonus;
        if (w > MAX_WEIGHT) w = MAX_WEIGHT;
        return w;
    }

    /**
     * Return the correct ModelType for a CapacityClass.
     */
    function modelTypeFor(capacityClass) {
        if (capacityClass === Weak) return Light;
        if (capacityClass === Medium) return MediumModel;
        return Heavy;
    }

    beforeEach(async function () {
        [owner, hospitalA, hospitalB, hospitalC, nonOwner] =
            await ethers.getSigners();

        const Factory = await ethers.getContractFactory("FLCoordinator");
        contract = await Factory.deploy();
        await contract.waitForDeployment();
    });

    // =========================================================================
    //  Registration
    // =========================================================================
    describe("Hospital Registration", function () {
        it("should register a hospital with Weak capacity and benchmark hash", async function () {
            await expect(
                contract.registerHospital(hospitalA.address, "City Hospital", Weak, benchmarkHash, await sigFor(hospitalA, benchmarkHash))
            )
                .to.emit(contract, "HospitalRegistered")
                .withArgs(hospitalA.address, "City Hospital", Weak, benchmarkHash);

            const info = await contract.getHospitalInfo(hospitalA.address);
            expect(info.name).to.equal("City Hospital");
            expect(info.capacityClass).to.equal(Weak);
            expect(info.isRegistered).to.be.true;
            expect(info.pocBenchmarkHash).to.equal(benchmarkHash);
            expect(info.roundsParticipated).to.equal(0);
        });

        it("should register hospitals with different capacity classes", async function () {
            await contract.registerHospital(hospitalA.address, "A", Weak, benchmarkHash, await sigFor(hospitalA, benchmarkHash));
            await contract.registerHospital(hospitalB.address, "B", Medium, benchmarkHash, await sigFor(hospitalB, benchmarkHash));
            await contract.registerHospital(hospitalC.address, "C", Strong, benchmarkHash, await sigFor(hospitalC, benchmarkHash));

            expect(await contract.getHospitalCount()).to.equal(3);

            expect((await contract.getHospitalInfo(hospitalA.address)).capacityClass)
                .to.equal(Weak);
            expect((await contract.getHospitalInfo(hospitalB.address)).capacityClass)
                .to.equal(Medium);
            expect((await contract.getHospitalInfo(hospitalC.address)).capacityClass)
                .to.equal(Strong);
        });

        it("should reject benchmark hash signed by the wrong hospital key", async function () {
            await expect(
                contract.registerHospital(
                    hospitalA.address,
                    "City Hospital",
                    Weak,
                    benchmarkHash,
                    await sigFor(hospitalB, benchmarkHash)
                )
            ).to.be.revertedWith("Invalid benchmark signature");
        });

        it("should reject duplicate registration", async function () {
            await contract.registerHospital(hospitalA.address, "A", Medium, benchmarkHash, await sigFor(hospitalA, benchmarkHash));
            await expect(
                contract.registerHospital(hospitalA.address, "A-dup", Strong, benchmarkHash, await sigFor(hospitalA, benchmarkHash))
            ).to.be.revertedWith("Hospital already registered");
        });

        it("should reject registration from non-owner", async function () {
            await expect(
                contract
                    .connect(nonOwner)
                    .registerHospital(hospitalA.address, "A", Medium, benchmarkHash, await sigFor(hospitalA, benchmarkHash))
            ).to.be.revertedWith("Not the contract owner");
        });

        it("should store distinct benchmark hashes per hospital", async function () {
            const hashA = ethers.keccak256(ethers.toUtf8Bytes("bench-A"));
            const hashB = ethers.keccak256(ethers.toUtf8Bytes("bench-B"));

            await contract.registerHospital(hospitalA.address, "A", Weak, hashA, await sigFor(hospitalA, hashA));
            await contract.registerHospital(hospitalB.address, "B", Strong, hashB, await sigFor(hospitalB, hashB));

            expect((await contract.getHospitalInfo(hospitalA.address)).pocBenchmarkHash)
                .to.equal(hashA);
            expect((await contract.getHospitalInfo(hospitalB.address)).pocBenchmarkHash)
                .to.equal(hashB);
        });
    });

    // =========================================================================
    //  Model Type Assignment
    // =========================================================================
    describe("Model Type Assignment", function () {
        it("should map Weak -> Light", async function () {
            expect(await contract.getModelType(Weak)).to.equal(Light);
        });

        it("should map Medium -> Medium", async function () {
            expect(await contract.getModelType(Medium)).to.equal(MediumModel);
        });

        it("should map Strong -> Heavy", async function () {
            expect(await contract.getModelType(Strong)).to.equal(Heavy);
        });

        it("should return assignedModelType in getHospitalInfo", async function () {
            await contract.registerHospital(hospitalA.address, "A", Weak, benchmarkHash, await sigFor(hospitalA, benchmarkHash));
            await contract.registerHospital(hospitalB.address, "B", Medium, benchmarkHash, await sigFor(hospitalB, benchmarkHash));
            await contract.registerHospital(hospitalC.address, "C", Strong, benchmarkHash, await sigFor(hospitalC, benchmarkHash));

            expect((await contract.getHospitalInfo(hospitalA.address)).assignedModelType).to.equal(Light);
            expect((await contract.getHospitalInfo(hospitalB.address)).assignedModelType).to.equal(MediumModel);
            expect((await contract.getHospitalInfo(hospitalC.address)).assignedModelType).to.equal(Heavy);
        });
    });

    // =========================================================================
    //  Round lifecycle & Submissions
    // =========================================================================
    describe("Round Lifecycle & Submissions", function () {
        const modelHash = ethers.keccak256(ethers.toUtf8Bytes("model-weights-v1"));

        beforeEach(async function () {
            await contract.registerHospital(hospitalA.address, "A", Strong, benchmarkHash, await sigFor(hospitalA, benchmarkHash));
            await contract.registerHospital(hospitalB.address, "B", Medium, benchmarkHash, await sigFor(hospitalB, benchmarkHash));
        });

        it("should start a new round", async function () {
            await expect(contract.startNewRound())
                .to.emit(contract, "RoundStarted")
                .withArgs(1);

            expect(await contract.currentRound()).to.equal(1);
        });

        it("should reject round start with no hospitals", async function () {
            const Factory = await ethers.getContractFactory("FLCoordinator");
            const empty = await Factory.deploy();
            await empty.waitForDeployment();

            await expect(empty.startNewRound()).to.be.revertedWith(
                "No hospitals registered"
            );
        });

        it("should accept a valid submission with correct model type", async function () {
            await contract.startNewRound();

            await expect(
                contract
                    .connect(hospitalA)
                    .submitUpdate(modelHash, 9500, 300, Heavy)
            )
                .to.emit(contract, "UpdateSubmitted")
                .withArgs(1, hospitalA.address, modelHash, 9500, 300, Heavy);

            const sub = await contract.getRoundSubmission(1, hospitalA.address);
            expect(sub.modelHash).to.equal(modelHash);
            expect(sub.confidence).to.equal(9500);
            expect(sub.ece).to.equal(300);
            expect(sub.timestamp).to.be.gt(0);
            expect(sub.modelType).to.equal(Heavy);
        });

        it("should reject submission with wrong model type", async function () {
            await contract.startNewRound();

            // hospitalA is Strong, assigned Heavy. Submitting Light should fail.
            await expect(
                contract.connect(hospitalA).submitUpdate(modelHash, 9500, 300, Light)
            ).to.be.revertedWith("Wrong model type for capacity class");

            // hospitalB is Medium, assigned MediumModel. Submitting Heavy should fail.
            await expect(
                contract.connect(hospitalB).submitUpdate(modelHash, 9500, 300, Heavy)
            ).to.be.revertedWith("Wrong model type for capacity class");
        });

        it("should update hospital reliability metrics after submission", async function () {
            await contract.startNewRound();
            await contract.connect(hospitalA).submitUpdate(modelHash, 8800, 200, Heavy);

            const info = await contract.getHospitalInfo(hospitalA.address);
            expect(info.confidence).to.equal(8800);
            expect(info.ece).to.equal(200);
        });

        it("should increment roundsParticipated after each submission", async function () {
            // Round 1
            await contract.startNewRound();
            await contract.connect(hospitalA).submitUpdate(modelHash, 9000, 300, Heavy);
            expect((await contract.getHospitalInfo(hospitalA.address)).roundsParticipated)
                .to.equal(1);

            // Round 2
            await contract.startNewRound();
            await contract.connect(hospitalA).submitUpdate(modelHash, 9100, 250, Heavy);
            expect((await contract.getHospitalInfo(hospitalA.address)).roundsParticipated)
                .to.equal(2);

            // Hospital B never submitted -> still 0
            expect((await contract.getHospitalInfo(hospitalB.address)).roundsParticipated)
                .to.equal(0);
        });

        it("should reject submission from unregistered address", async function () {
            await contract.startNewRound();
            await expect(
                contract.connect(nonOwner).submitUpdate(modelHash, 9000, 500, Light)
            ).to.be.revertedWith("Not a registered hospital");
        });

        it("should reject submission before any round starts", async function () {
            await expect(
                contract.connect(hospitalA).submitUpdate(modelHash, 9000, 500, Heavy)
            ).to.be.revertedWith("No active round");
        });

        it("should reject duplicate submission in the same round", async function () {
            await contract.startNewRound();
            await contract.connect(hospitalA).submitUpdate(modelHash, 9000, 500, Heavy);

            await expect(
                contract.connect(hospitalA).submitUpdate(modelHash, 9200, 400, Heavy)
            ).to.be.revertedWith("Already submitted this round");
        });

        it("should reject confidence > SCALE", async function () {
            await contract.startNewRound();
            await expect(
                contract.connect(hospitalA).submitUpdate(modelHash, 10_001, 500, Heavy)
            ).to.be.revertedWith("Confidence out of range");
        });

        it("should reject ECE > SCALE", async function () {
            await contract.startNewRound();
            await expect(
                contract.connect(hospitalA).submitUpdate(modelHash, 9000, 10_001, Heavy)
            ).to.be.revertedWith("ECE out of range");
        });

        it("should track round submitter count", async function () {
            await contract.startNewRound();
            await contract.connect(hospitalA).submitUpdate(modelHash, 9000, 300, Heavy);
            await contract.connect(hospitalB).submitUpdate(modelHash, 8500, 400, MediumModel);

            expect(await contract.getRoundSubmitterCount(1)).to.equal(2);
        });
    });

    // =========================================================================
    //  Ensemble Prediction Recording
    // =========================================================================
    describe("Ensemble Prediction Recording", function () {
        const modelHash = ethers.keccak256(ethers.toUtf8Bytes("model-v1"));
        const predHash = ethers.keccak256(ethers.toUtf8Bytes("ensemble-pred-r1"));

        beforeEach(async function () {
            await contract.registerHospital(hospitalA.address, "A", Strong, benchmarkHash, await sigFor(hospitalA, benchmarkHash));
            await contract.registerHospital(hospitalB.address, "B", Medium, benchmarkHash, await sigFor(hospitalB, benchmarkHash));
            await contract.startNewRound();
            await contract.connect(hospitalA).submitUpdate(modelHash, 9000, 300, Heavy);
            await contract.connect(hospitalB).submitUpdate(modelHash, 8500, 400, MediumModel);
        });

        it("should record an ensemble prediction", async function () {
            await expect(
                contract.recordEnsemblePrediction(predHash, 2)
            )
                .to.emit(contract, "EnsemblePredictionRecorded")
                .withArgs(1, predHash, 2);

            const record = await contract.getEnsembleRecord(1);
            expect(record.predictionHash).to.equal(predHash);
            expect(record.participantCount).to.equal(2);
            expect(record.timestamp).to.be.gt(0);
        });

        it("should reject duplicate ensemble recording in same round", async function () {
            await contract.recordEnsemblePrediction(predHash, 2);
            await expect(
                contract.recordEnsemblePrediction(predHash, 2)
            ).to.be.revertedWith("Ensemble already recorded for this round");
        });

        it("should reject ensemble recording with 0 participants", async function () {
            await expect(
                contract.recordEnsemblePrediction(predHash, 0)
            ).to.be.revertedWith("No participants");
        });

        it("should reject ensemble recording from non-owner", async function () {
            await expect(
                contract.connect(hospitalA).recordEnsemblePrediction(predHash, 2)
            ).to.be.revertedWith("Not the contract owner");
        });

        it("should reject ensemble recording before any round", async function () {
            const Factory = await ethers.getContractFactory("FLCoordinator");
            const fresh = await Factory.deploy();
            await fresh.waitForDeployment();
            await expect(
                fresh.recordEnsemblePrediction(predHash, 1)
            ).to.be.revertedWith("No active round");
        });

        it("should allow separate recordings per round", async function () {
            await contract.recordEnsemblePrediction(predHash, 2);

            // Start round 2
            await contract.startNewRound();
            await contract.connect(hospitalA).submitUpdate(modelHash, 9200, 250, Heavy);

            const predHash2 = ethers.keccak256(ethers.toUtf8Bytes("ensemble-pred-r2"));
            await contract.recordEnsemblePrediction(predHash2, 1);

            const r1 = await contract.getEnsembleRecord(1);
            const r2 = await contract.getEnsembleRecord(2);
            expect(r1.predictionHash).to.equal(predHash);
            expect(r2.predictionHash).to.equal(predHash2);
            expect(r1.participantCount).to.equal(2);
            expect(r2.participantCount).to.equal(1);
        });
    });

    // =========================================================================
    //  Weight Calculations (core focus)
    // =========================================================================
    describe("Weight Calculations", function () {

        // -- calculateWeightPure (stateless, any inputs) ----------------------
        describe("calculateWeightPure (no participation bonus)", function () {

            it("Strong / conf=95% / ece=5% / 0 rounds -> 10830", async function () {
                const w = await contract.calculateWeightPure(Strong, 9500, 500, 0);
                expect(w).to.equal(expectedWeight(CAP_STRONG, 9500n, 500n, 0n));
                expect(w).to.equal(10_830n);
            });

            it("Medium / conf=90% / ece=10% / 0 rounds -> 8100", async function () {
                const w = await contract.calculateWeightPure(Medium, 9000, 1000, 0);
                expect(w).to.equal(expectedWeight(CAP_MEDIUM, 9000n, 1000n, 0n));
                expect(w).to.equal(8_100n);
            });

            it("Weak / conf=80% / ece=20% / 0 rounds -> 5120", async function () {
                const w = await contract.calculateWeightPure(Weak, 8000, 2000, 0);
                expect(w).to.equal(expectedWeight(CAP_WEAK, 8000n, 2000n, 0n));
                expect(w).to.equal(5_120n);
            });

            it("Strong / conf=100% / ece=0% / 0 rounds -> 12000", async function () {
                const w = await contract.calculateWeightPure(Strong, 10000, 0, 0);
                expect(w).to.equal(12_000n);
            });

            it("conf=0% -> weight is 0 regardless of class", async function () {
                expect(await contract.calculateWeightPure(Strong, 0, 0, 0)).to.equal(0);
                expect(await contract.calculateWeightPure(Medium, 0, 5000, 0)).to.equal(0);
            });

            it("ece=100% -> weight is 0 regardless of class", async function () {
                expect(await contract.calculateWeightPure(Strong, 10000, 10000, 0)).to.equal(0);
                expect(await contract.calculateWeightPure(Weak, 10000, 10000, 0)).to.equal(0);
            });

            it("should revert on confidence > SCALE", async function () {
                await expect(
                    contract.calculateWeightPure(Medium, 10_001, 0, 0)
                ).to.be.revertedWith("Confidence out of range");
            });

            it("should revert on ECE > SCALE", async function () {
                await expect(
                    contract.calculateWeightPure(Medium, 5000, 10_001, 0)
                ).to.be.revertedWith("ECE out of range");
            });
        });

        // -- calculateWeightPure WITH participation bonus ---------------------
        describe("calculateWeightPure (with participation bonus)", function () {

            it("Strong / conf=95% / ece=5% / 1 round -> 10830 + 500 = 11330", async function () {
                const w = await contract.calculateWeightPure(Strong, 9500, 500, 1);
                expect(w).to.equal(10_830n + 500n);
            });

            it("Medium / conf=90% / ece=10% / 3 rounds -> 8100 + 1500 = 9600", async function () {
                const w = await contract.calculateWeightPure(Medium, 9000, 1000, 3);
                expect(w).to.equal(8_100n + 1_500n);
            });

            it("Participation bonus caps at 2500 (5 rounds)", async function () {
                const w5 = await contract.calculateWeightPure(Medium, 9000, 1000, 5);
                const w10 = await contract.calculateWeightPure(Medium, 9000, 1000, 10);
                expect(w5).to.equal(8_100n + 2_500n);
                expect(w10).to.equal(8_100n + 2_500n);
                expect(w5).to.equal(w10);
            });

            it("Participation bonus cannot push weight above MAX_WEIGHT", async function () {
                const w = await contract.calculateWeightPure(Strong, 10000, 0, 5);
                expect(w).to.equal(12_000n + 2_500n);
            });

            it("Bonus helps low-capacity nodes", async function () {
                const w = await contract.calculateWeightPure(Weak, 10000, 0, 5);
                expect(w).to.equal(8_000n + 2_500n);
            });

            it("Zero metrics + participation = only bonus", async function () {
                const w = await contract.calculateWeightPure(Strong, 0, 0, 2);
                expect(w).to.equal(1_000n);
            });
        });

        // -- calculateWeight (reads stored hospital state) --------------------
        describe("calculateWeight (from stored metrics)", function () {
            const modelHash = ethers.keccak256(ethers.toUtf8Bytes("weights-v1"));

            beforeEach(async function () {
                await contract.registerHospital(hospitalA.address, "A", Strong, benchmarkHash, await sigFor(hospitalA, benchmarkHash));
                await contract.registerHospital(hospitalB.address, "B", Medium, benchmarkHash, await sigFor(hospitalB, benchmarkHash));
                await contract.registerHospital(hospitalC.address, "C", Weak, benchmarkHash, await sigFor(hospitalC, benchmarkHash));
            });

            it("should return 0 before any submission (metrics default to 0)", async function () {
                const w = await contract.calculateWeight(hospitalA.address);
                expect(w).to.equal(0);
            });

            it("should include participation bonus after submission", async function () {
                await contract.startNewRound();
                await contract
                    .connect(hospitalA)
                    .submitUpdate(modelHash, 9500, 500, Heavy); // Strong, 95%, 5%, 1 round

                const w = await contract.calculateWeight(hospitalA.address);
                // 10830 base + 500 bonus = 11330
                expect(w).to.equal(11_330n);
            });

            it("should increase weight across multiple rounds", async function () {
                // Round 1
                await contract.startNewRound();
                await contract.connect(hospitalA).submitUpdate(modelHash, 9000, 1000, Heavy);
                const w1 = await contract.calculateWeight(hospitalA.address);

                // Round 2 (same metrics)
                await contract.startNewRound();
                await contract.connect(hospitalA).submitUpdate(modelHash, 9000, 1000, Heavy);
                const w2 = await contract.calculateWeight(hospitalA.address);

                // w2 should be 500 more than w1 (one extra round)
                expect(w2 - w1).to.equal(500n);
            });

            it("should reflect per-hospital capacity class in the weight", async function () {
                await contract.startNewRound();

                // All submit identical reliability metrics, each with correct model type
                await contract.connect(hospitalA).submitUpdate(modelHash, 9000, 1000, Heavy);
                await contract.connect(hospitalB).submitUpdate(modelHash, 9000, 1000, MediumModel);
                await contract.connect(hospitalC).submitUpdate(modelHash, 9000, 1000, Light);

                const wA = await contract.calculateWeight(hospitalA.address); // Strong
                const wB = await contract.calculateWeight(hospitalB.address); // Medium
                const wC = await contract.calculateWeight(hospitalC.address); // Weak

                // Strong > Medium > Weak (all have same 1-round bonus of 500)
                expect(wA).to.be.gt(wB);
                expect(wB).to.be.gt(wC);

                // Strong: 9720 + 500 = 10220
                // Medium: 8100 + 500 = 8600
                // Weak: 6480 + 500 = 6980
                expect(wA).to.equal(10_220n);
                expect(wB).to.equal(8_600n);
                expect(wC).to.equal(6_980n);
            });

            it("should revert for unregistered address", async function () {
                await expect(
                    contract.calculateWeight(nonOwner.address)
                ).to.be.revertedWith("Hospital not registered");
            });
        });
    });

    // =========================================================================
    //  Gas Benchmarking
    // =========================================================================
    describe("Gas Benchmarking", function () {
        const modelHash = ethers.keccak256(ethers.toUtf8Bytes("model-gas-test"));
        const predHash = ethers.keccak256(ethers.toUtf8Bytes("ensemble-gas-test"));

        it("should measure gas for registerHospital", async function () {
            const tx = await contract.registerHospital(
                hospitalA.address, "Gas Test Hospital", Strong,
                benchmarkHash, await sigFor(hospitalA, benchmarkHash)
            );
            const receipt = await tx.wait();
            console.log(`      registerHospital gas: ${receipt.gasUsed}`);
            expect(receipt.gasUsed).to.be.gt(0);
            expect(receipt.gasUsed).to.be.lt(300_000);
        });

        it("should measure gas for submitUpdate", async function () {
            await contract.registerHospital(
                hospitalA.address, "A", Strong,
                benchmarkHash, await sigFor(hospitalA, benchmarkHash)
            );
            await contract.startNewRound();

            const tx = await contract
                .connect(hospitalA)
                .submitUpdate(modelHash, 9500, 500, Heavy);
            const receipt = await tx.wait();
            console.log(`      submitUpdate gas: ${receipt.gasUsed}`);
            expect(receipt.gasUsed).to.be.gt(0);
            expect(receipt.gasUsed).to.be.lt(300_000);
        });

        it("should measure gas for startNewRound", async function () {
            await contract.registerHospital(
                hospitalA.address, "A", Medium,
                benchmarkHash, await sigFor(hospitalA, benchmarkHash)
            );

            const tx = await contract.startNewRound();
            const receipt = await tx.wait();
            console.log(`      startNewRound gas: ${receipt.gasUsed}`);
            expect(receipt.gasUsed).to.be.gt(0);
            expect(receipt.gasUsed).to.be.lt(100_000);
        });

        it("should measure gas for recordEnsemblePrediction", async function () {
            await contract.registerHospital(
                hospitalA.address, "A", Strong,
                benchmarkHash, await sigFor(hospitalA, benchmarkHash)
            );
            await contract.startNewRound();
            await contract.connect(hospitalA).submitUpdate(modelHash, 9000, 300, Heavy);

            const tx = await contract.recordEnsemblePrediction(predHash, 1);
            const receipt = await tx.wait();
            console.log(`      recordEnsemblePrediction gas: ${receipt.gasUsed}`);
            expect(receipt.gasUsed).to.be.gt(0);
            expect(receipt.gasUsed).to.be.lt(150_000);
        });
    });
});
