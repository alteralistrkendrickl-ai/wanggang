"""Identity/domain-balanced batches for strict WiSig A1 experiments."""

import math

import numpy as np
from torch.utils.data import Sampler


class IdentityDomainBatchSampler(Sampler):
    """Sample identities, distinct domains, then samples within each domain."""

    def __init__(self, labels, domains, identities_per_batch=8,
                 domains_per_identity=2, samples_per_domain=2, seed=2024,
                 batches_per_epoch=None):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.domains = np.asarray(domains, dtype=np.int64)
        if self.labels.shape != self.domains.shape or self.labels.ndim != 1:
            raise ValueError("labels and domains must be aligned one-dimensional arrays")
        self.identities_per_batch = int(identities_per_batch)
        self.domains_per_identity = int(domains_per_identity)
        self.samples_per_domain = int(samples_per_domain)
        if min(self.identities_per_batch, self.domains_per_identity,
               self.samples_per_domain) < 1:
            raise ValueError("A1 batch dimensions must be positive")
        if self.domains_per_identity < 2:
            raise ValueError("A1 requires at least two distinct domains per identity")
        self.batch_size = (self.identities_per_batch * self.domains_per_identity
                           * self.samples_per_domain)
        self.seed = int(seed)
        self.epoch = 0
        self.cells = {}
        valid_identities = []
        for identity in sorted(np.unique(self.labels).astype(int).tolist()):
            identity_positions = np.flatnonzero(self.labels == identity)
            cells = {}
            for domain in sorted(np.unique(self.domains[identity_positions]).astype(int).tolist()):
                positions = identity_positions[self.domains[identity_positions] == domain]
                if len(positions) >= self.samples_per_domain:
                    cells[domain] = positions
            if len(cells) >= self.domains_per_identity:
                valid_identities.append(identity)
                self.cells[identity] = cells
        self.identities = np.asarray(valid_identities, dtype=np.int64)
        missing = sorted(set(np.unique(self.labels).astype(int)) - set(valid_identities))
        if missing:
            raise ValueError(
                "Identities cannot form requested cross-domain batch cells: "
                f"{missing}"
            )
        if len(self.identities) < self.identities_per_batch:
            raise ValueError("Not enough identities for one A1 batch")
        default_batches = math.ceil(len(self.labels) / self.batch_size)
        self.batches_per_epoch = default_batches if batches_per_epoch is None else int(batches_per_epoch)
        if self.batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be positive")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003)
        for _ in range(self.batches_per_epoch):
            identities = rng.choice(self.identities, self.identities_per_batch, replace=False)
            batch = []
            for identity in identities:
                cells = self.cells[int(identity)]
                chosen_domains = rng.choice(
                    np.asarray(list(cells), dtype=np.int64),
                    self.domains_per_identity,
                    replace=False,
                )
                for domain in chosen_domains:
                    positions = rng.choice(
                        cells[int(domain)], self.samples_per_domain, replace=False
                    )
                    batch.extend(positions.astype(int).tolist())
            yield batch
