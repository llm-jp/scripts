"""
test p2p comm
"""
import os
import socket
import sys
import torch

def p2p(sender, receiver):
    """
    main
    """
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    lank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    host = socket.gethostname()
    print(f"{host} [{rank}/{world_size}]", flush=True)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == sender:
        s = torch.tensor([rank * world_size + receiver + 123], device=f"cuda:{lank}")
        print(f"{host} [{rank}] : BEG send {rank} -> {receiver} {s[0]}", flush=True)
        torch.distributed.send(s, receiver)
        print(f"{host} [{rank}] : END send {rank} -> {receiver} {s[0]}", flush=True)
    elif rank == receiver:
        r = torch.tensor([0], device=f"cuda:{lank}")
        print(f"{host} [{rank}] : BEG recv", flush=True)
        sender_ = torch.distributed.recv(r, sender)
        print(f"{host} [{rank}] : END recv {rank} <- {sender} {r[0]}", flush=True)
        assert(sender_ == sender), (sender_, sender)
        assert(r[0] == sender * world_size + rank + 123), (r[0], sender * world_size + rank + 123)
    print(f"{host} [{rank}] : BEG barrier", flush=True)
    torch.distributed.barrier()
    print(f"{host} [{rank}] : END barrier", flush=True)
    if rank == receiver:
        print(f"{host} [{rank}] : OK {sender} -> {receiver}", flush=True)

def main():
    sender = int(sys.argv[1])
    receiver = int(sys.argv[2])
    p2p(sender, receiver)

main()