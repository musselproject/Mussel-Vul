from llmtuner.train.tuner import run_exp
import wandb

def main():
    # wandb.login()
    run_exp()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()
