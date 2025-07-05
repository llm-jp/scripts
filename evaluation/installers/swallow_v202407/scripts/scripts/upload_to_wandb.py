# %%
import wandb
import pandas as pd
import json
import argparse

# %%
import argparse

# Create a parser
parser = argparse.ArgumentParser(description="")

# Add arguments
parser.add_argument("--entity", type=str)
parser.add_argument("--project", type=str)
parser.add_argument("--run", type=str)
parser.add_argument("--aggregated_result", type=str)

# Parse arguments
args = parser.parse_args()


# %%
wandb.init(entity = f"{args.entity}", project=f"{args.project}", name=f"{args.run}")


# %%
# aggregated_result_fn = f"/home/mdxuser/projects/environ/swallow/environment/src/swallow-evaluation/results/tokyotech-llm/Swallow-7b-instruct-v0.1/aggregated_result.json"
with open(args.aggregated_result, "rb")  as f:
    results = json.load(f)

# %%
task_to_drop = [i==-1.0 for i in map(float, results["overall"].split(","))]
# print(task_to_drop)
tasks = [t for t, d in zip(results["tasks"], task_to_drop) if not d]
overall = [s for s, d in zip(map(float, results["overall"].split(",")), task_to_drop) if not d]

# %%
def wandb_log(wandb, df, tbl_name):
    tbl = wandb.Table(columns=df.columns.tolist(), data=df.to_numpy().tolist())
    wandb.log({f"{tbl_name}": tbl})
    

# %%
radar_df = pd.DataFrame({
    "category": tasks,
    "score": overall
})
wandb_log(wandb, radar_df, "radar_table")

# %%
model_cmp_df = pd.DataFrame({"model": results["model"]}| {k: v for k, v in  results["result"].items() if v!=-1}, index=[0])
wandb_log(wandb, model_cmp_df, "model_comparison_table")

# %%
wandb.finish()
