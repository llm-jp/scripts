# P2P Test

Tests the send `torch.distributed.send` and receive `torch.distributed.recv` of a tensor between two specified processes.

## How to use

Create an `outputs` directory in the same hierarchy as the `scripts`.  
Submit a job by specifying 
- partition name (here, `--partition=gpu-debug`)
- number of nodes (here, `--nodes=2`)
- ranks of the two processes to be tested (here, `0` and `9`).

```
sbatch --partition=gpu-debug --nodes=2 scripts/server/gpu/p2p_test/sbatch.sh 0 9
```

## Example output

If the communication between the processes fails, the following error will be displayed.

```
[rank9]:[E ProcessGroupNCCL.cpp:577] [Rank 9] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank9]:[E ProcessGroupNCCL.cpp:583] [Rank 9] To avoid data inconsistency, we are taking the entire process down.
[rank9]:[E ProcessGroupNCCL.cpp:1414] [PG 0 Rank 9] Process group watchdog thread terminated with exception: NCCL error: remote process exited or there was a network error, NCCL version 2.20.5
ncclRemoteError: A call failed possibly due to a network error or a remote process exiting prematurely.
Last error:
NET/IB : Got completion from peer 10.5.0.34<33956> with error 5, opcode 0, len 1, vendor err 244 (Recv) localGid ::ffff:10.6.0.35 remoteGid ::ffff:10.5.0.34
Exception raised from checkForNCCLErrorsInternal at ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1723 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x57 (0x14fc4da49897 in /home/shared/experiments/0015/p2p-test/environment/venv/lib/python3.10/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::checkForNCCLErrorsInternal(std::shared_ptr<c10d::NCCLComm>&) + 0x220 (0x14fc4ec775f0 in /home/shared/experiments/0015/p2p-test/environment/venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::checkAndSetException() + 0x7c (0x14fc4ec7783c in /home/shared/experiments/0015/p2p-test/environment/venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::watchdogHandler() + 0x180 (0x14fc4ec7ca60 in /home/shared/experiments/0015/p2p-test/environment/venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x10c (0x14fc4ec7ddcc in /home/shared/experiments/0015/p2p-test/environment/venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
frame #5: <unknown function> + 0xdbad4 (0x14fc9a6dbad4 in /lib64/libstdc++.so.6)
frame #6: <unknown function> + 0x89c02 (0x14fc9b689c02 in /lib64/libc.so.6)
frame #7: <unknown function> + 0x10ec40 (0x14fc9b70ec40 in /lib64/libc.so.6)
...
```