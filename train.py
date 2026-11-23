import hydra
from omegaconf import DictConfig

@hydra.main(
    version_base=None,
    config_path="src/dynamics/config/train/model",
    config_name="wm"
)
def train(cfg: DictConfig):
    print(cfg.learning_rate)
    print(cfg.batch_size)
    print(cfg.epochs)

if __name__ == "__main__":
    train()