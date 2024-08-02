"""Submit a model as a W&B artifact."""
import wandb
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--entity",
    type=str,
    required=True
)
parser.add_argument(
    "--project",
    type=str,
    required=True
)
parser.add_argument(
    "--model_path",
    type=str,
    required=True
)
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
)
parser.add_argument(
    "-v",
    "--model_version",
    type=str,
    required=True
)
args = parser.parse_args()

with wandb.init(entity=args.entity, project=args.project, job_type="upload_data") as run:
    model_artifact = wandb.Artifact(
        name=args.model_name,
        type="model", 
        metadata={"version": args.model_version},
        description=f"{args.model_name} (version: {args.model_version})"
    )
    model_artifact.add_dir(args.model_path, skip_cache=True, policy="immutable")
    run.log_artifact(model_artifact)
