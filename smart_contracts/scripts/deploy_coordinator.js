const { ethers } = require("hardhat");

async function main() {
  console.log("Deploying FLCoordinator contract...");

  const FLCoordinator = await ethers.getContractFactory("FLCoordinator");
  const contract = await FLCoordinator.deploy();
  await contract.waitForDeployment();

  const address = await contract.getAddress();
  console.log(`FLCoordinator deployed to: ${address}`);

  return address;
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
