import os
import time
import pandas as pd
import pickle
import numpy as np
from rdkit import Chem
import torch
import torch.nn.functional as F
from torch.nn import (
    ModuleList,
    Sequential,
    Embedding,
    Linear,
    BatchNorm1d,
    ReLU,
    Dropout,
)
import argparse
from torch import Tensor
from torch_sparse import SparseTensor
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from torch_geometric import seed_everything
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.data import Batch
from molgnn.metrics import MeanSquaredError, R2Score, MeanAbsoluteError
from molgnn.datasets import MNET
from gg.ga import Genetic
from gg.moves import Crossover, Mutate
from gg.score import PytorchScoringFunction

# Ax optimized model architecture
class GINERegressor(LightningModule):
    def __init__(
        self,
        num_tasks: int,
        transform: object,
        norm: bool = True,
        conv_layers: int = 8,
        conv_dim: int = 209,
        mlp_layers: int = 1,
        mlp_dim: int = 104,
        dropout: float = 0.2488482220343721,
        learning_rate: float = 0.001432167801777256,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_tasks = num_tasks
        self.transform = transform
        self.embedding = Embedding(100, conv_dim)
        self.edge_embedding = Embedding(4, conv_dim)

        self.convs = ModuleList()
        for _ in range(conv_layers):
            mlp = Sequential(
                Linear(conv_dim, 2 * conv_dim),
                BatchNorm1d(2 * conv_dim),
                ReLU(),
                Linear(2 * conv_dim, conv_dim),
                BatchNorm1d(conv_dim),
                ReLU(),
            )
            conv = GINEConv(mlp, train_eps=True)
            self.convs.append(conv)
            in_channels = conv_dim

        self.norms = None
        if norm:
            self.norms = ModuleList()
            for _ in range(conv_layers):
                self.norms.append(BatchNorm1d(conv_dim))

        fc = []
        for i in range(mlp_layers - 1):
            fc += [
                Sequential(
                    Linear(in_channels, mlp_dim),
                    BatchNorm1d(mlp_dim),
                    ReLU(),
                    Dropout(p=dropout),
                )
            ]
            in_channels = mlp_dim
        fc += [Linear(in_channels, num_tasks)]
        self.regressor = ModuleList(fc)
        self.lr = learning_rate

        self.val_r2 = R2Score(num_outputs=num_tasks)
        self.test_r2 = R2Score(num_outputs=num_tasks)
        self.val_r2_rv = R2Score(num_outputs=num_tasks, multioutput="raw_values")
        self.test_r2_rv = R2Score(num_outputs=num_tasks, multioutput="raw_values")
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

    def forward(
        self, x: Tensor, edge_index: SparseTensor, edge_attr: Tensor, batch: Tensor
    ) -> Tensor:
        x = self.embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        if self.norms is not None:
            for bn, conv in zip(self.norms, self.convs):
                x = F.relu(bn(conv(x, edge_index, edge_attr)))
        else:
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_attr))
        x = global_add_pool(x, batch)
        for nn in self.regressor:
            x = nn(x)
        return x

    def training_step(self, batch: Batch, batch_idx: int):
        y_hat = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        train_loss = F.mse_loss(y_hat, batch.y)
        return train_loss

    def validation_step(self, batch: Batch, batch_idx: int):
        y = self.transform.inverse_transform(batch.y)
        y_hat = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_hat = self.transform.inverse_transform(y_hat)
        self.val_r2(y_hat, y)
        self.val_r2_rv(y_hat, y)
        self.val_mse(y_hat, y)
        self.val_mae(y_hat, y)
        self.log(
            "val_r2",
            self.val_r2,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=55,
        )
        self.log(
            "val_r2_rv",
            self.val_r2_rv,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=False,
            batch_size=55,
        )
        self.log(
            "val_mse",
            self.val_mse,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=55,
        )
        self.log(
            "val_mae",
            self.val_mae,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=55,
        )

    def test_step(self, batch: Batch, batch_idx: int):
        y = self.transform.inverse_transform(batch.y)
        y_hat = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_hat = self.transform.inverse_transform(y_hat)
        self.test_r2(y_hat, y)
        self.test_r2_rv(y_hat, y)
        self.test_mse(y_hat, y)
        self.test_mae(y_hat, y)
        self.log(
            "test_r2",
            self.test_r2,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=55,
        )
        self.log(
            "test_r2_rv",
            self.test_r2_rv,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=False,
            batch_size=55,
        )
        self.log(
            "test_mse",
            self.test_mse,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=55,
        )
        self.log(
            "test_mae",
            self.test_mae,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=55,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# Train Model
if __name__ == "__main__":

    seed_everything(123)

    # read dataset name
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="moleculenet dataset")
    a = parser.parse_args()

    # load datamodule
    path = os.path.join(os.path.expanduser("~"), "data", "mnet")
    datamodule = MNET(path, name=a.dataset)

    # Select task
    datamodule = MNET(path, name=a.dataset, taskid=9)
    model = GINERegressor(datamodule.num_tasks, datamodule.transform)
    checkpoint_callback = ModelCheckpoint(monitor="val_r2", mode="max", save_top_k=1)
    trainer = Trainer(gpus=1, max_epochs=25, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=datamodule)
    val_score_dict = trainer.validate(datamodule=datamodule, ckpt_path="best")[0]
    print(val_score_dict)

    # Graph generation
    filename = os.path.join(os.getcwd(), "data.csv")
    params = {
        "generations": 50,
        "population_size": 200,
        "mating_pool_size": 200,
        "prune_population": True,
        "max_score": 20.0,
        "filename": filename,
    }

    co = Crossover(n_tries=10)
    mu = Mutate(rate=0.01, n_tries=10, size_mean=39.15, size_std=3.50)
    sc = PytorchScoringFunction(filename, model, yt, sa_score=False, cycle_score=False)
    ga = Genetic(co, mu, sc, params)

    t0 = time.time()
    results, population, generations_list = ga()
    t1 = time.time()

    # Create file path?
    base_dir = os.path.basename(os.getcwd())
    time = time.time()
    if not os.path.exists("csv"):
        os.mkdir("csv")

    filename = "csv/{}_{}.csv".format(base_dir, time)

    # Create a list of generated results
    unique_population, unique_results = [], []
    i = 0
    for pop, res in zip(population, results):
        sml = Chem.MolToSmiles(population[i])
        unique_population.append(sml)
        unique_results.append(results[i])
        i += 1

    df = pd.DataFrame.from_dict(
        {"population": unique_population, "results": unique_results}
    )
    df.to_csv(filename)

    # Evaluate results
    print(results[0], Chem.MolToSmiles(population[0]))
    print(
        "max score {}, mean {} +/- {}".format(
            max(results), np.array(results).mean(), np.array(results).std()
        )
    )
    print(
        "mean generations {} +/- {}".format(
            np.array(generations_list).mean(), np.array(generations_list).std()
        )
    )
    print("time {} minutes".format((t1 - t0) / 60.0))
