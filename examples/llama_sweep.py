#!/usr/bin/python

from dataclasses import dataclass, field

import os
import pty
import subprocess
import threading
import time
import typing


@dataclass
class Job:
    id: int
    done: bool = False
    status: int = -1


# Run N instances of the distill llama trainer simultaneously on 4
# TPU VMs. Change the data_seed between each run. Each of the 4 runs
# should be in a separate thread.
def run_jobs(tpu_id: int, jobs: typing.List[Job]):
    # spawn the job using pty, and collect the output of stdout and stderr into buffers.
    for job in jobs:
        with open(f"logs/tpu.{tpu_id}.log", "w") as f:

            def _read(fd):
                buf = os.read(fd, 1024)
                f.write(buf.decode("utf-8"))
                return buf

            wait_code = pty.spawn(
                [
                    "python",
                    "infra/launch.py",
                    "--foreground",
                    f"--tpu_name=tpu-{tpu_id}",
                    "--",
                    "python",
                    "examples/gsm8k-lora/gsm8k_lora.py",
                    "--config=examples/gsm8k-lora/gsm8k-llama2.yaml",
                    "--data_cache_dir=gs://wasabi-tpu-training/gsm8k/data",
                    f"--trainer.seed={job.id + 100}",
                    f"--trainer.checkpointer.base_path=gs://wasabi-tpu-training/llama3-gsm8k/job-{job.id}",
                    f"--hf_save_path=gs://wasabi-tpu-training/llama3-gsm8k/hf-chkpt-{job.id}",
                    f"--data_seed={job.id}",
                ],
                _read,
            )
            exit_code = os.waitstatus_to_exitcode(wait_code)
            print("Done:", exit_code)
            job.status = exit_code
            job.done = True


def main():
    jobs = [Job(id=i) for i in range(16)]
    threads = []
    num_tpus = 4
    for tpu_id in range(num_tpus):
        tpu_jobs = [job for (i, job) in enumerate(jobs) if i % num_tpus == tpu_id]
        threads.append(threading.Thread(target=run_jobs, args=(tpu_id, tpu_jobs)))
        threads[-1].start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
