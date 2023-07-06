import hydra
from omegaconf import DictConfig

from src.trainer import Trainer


@hydra.main(config_path=r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl_iris\config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
