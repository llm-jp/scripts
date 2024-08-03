"""Submit a model as a W&B artifact."""
import wandb
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--entity",
    type=str,
    required=True,
    help="The W&B entity to which the model will be uploaded.",
)
parser.add_argument(
    "--project",
    type=str,
    required=True,
    help="The W&B project to which the model will be uploaded.",
)
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="The path to the model directory to be uploaded.",
)
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="The name of the model to be uploaded.",
)
parser.add_argument(
    "-v",
    "--model_version",
    type=str,
    required=True,
    help="The version of the model to be uploaded.",
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
