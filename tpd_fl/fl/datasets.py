"""
Synthetic PII Dataset and Partitioning Utilities for TPD+FL.

Provides tools for generating synthetic training data with embedded
PII (Personally Identifiable Information) and partitioning it across
FL clients under various heterogeneity regimes:

- **IID partitioning**: uniform random split across clients.
- **Non-IID domain-skewed**: each client has a skewed distribution
  over text domains (medical, financial, legal, etc.).
- **Non-IID sensitivity-skewed**: each client specialises in
  different PII patterns (names, emails, phone numbers, etc.).

The synthetic dataset uses configurable templates with slot-based
PII injection, making it fully reproducible via seeds and suitable
for privacy research where real PII data cannot be used.

Usage::

    from tpd_fl.fl.datasets import (
        SyntheticPIIDataset,
        partition_iid,
        partition_non_iid_domain,
        partition_non_iid_sensitivity,
    )

    dataset = SyntheticPIIDataset(num_samples=1000, seed=42)
    samples = dataset.generate()
    client_datasets = partition_iid(samples, num_clients=10)

This module depends on the Python standard library only (no torch
required for generation, but samples contain token IDs as lists).
"""

from __future__ import annotations

import hashlib
import math
import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# PII Generators
# ---------------------------------------------------------------------------

# Name pools — synthetic, non-real
_FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karl", "Laura", "Mallory", "Nora", "Oscar", "Peggy",
    "Quinn", "Rupert", "Sybil", "Trent", "Uma", "Victor", "Wendy", "Xander",
    "Yolanda", "Zach", "Aria", "Blake", "Cora", "Derek", "Elise", "Felix",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young",
]

_DOMAINS = [
    "example.com", "testmail.org", "sample.net", "fakecorp.io",
    "synth-email.dev", "notreal.edu", "placeholder.co", "demo-corp.com",
]

_STREET_NAMES = [
    "Main St", "Oak Ave", "Elm Blvd", "Maple Dr", "Cedar Ln",
    "Pine Rd", "Birch Ct", "Spruce Way", "Willow Pl", "Ash Dr",
]

_CITIES = [
    "Springfield", "Shelbyville", "Centerville", "Fairview", "Georgetown",
    "Greenville", "Madison", "Franklin", "Clinton", "Arlington",
]

_STATES = [
    "CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
]


class _PIIGenerator:
    """Deterministic PII value generator seeded by a random.Random instance."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def name(self) -> str:
        first = self._rng.choice(_FIRST_NAMES)
        last = self._rng.choice(_LAST_NAMES)
        return f"{first} {last}"

    def email(self) -> str:
        first = self._rng.choice(_FIRST_NAMES).lower()
        last = self._rng.choice(_LAST_NAMES).lower()
        domain = self._rng.choice(_DOMAINS)
        num = self._rng.randint(1, 999)
        return f"{first}.{last}{num}@{domain}"

    def phone(self) -> str:
        area = self._rng.randint(200, 999)
        mid = self._rng.randint(200, 999)
        last = self._rng.randint(1000, 9999)
        fmt = self._rng.choice([
            f"({area}) {mid}-{last}",
            f"{area}-{mid}-{last}",
            f"+1-{area}-{mid}-{last}",
            f"{area}.{mid}.{last}",
        ])
        return fmt

    def ssn(self) -> str:
        a = self._rng.randint(100, 999)
        b = self._rng.randint(10, 99)
        c = self._rng.randint(1000, 9999)
        return f"{a}-{b}-{c}"

    def credit_card(self) -> str:
        groups = [
            str(self._rng.randint(1000, 9999)) for _ in range(4)
        ]
        return " ".join(groups)

    def address(self) -> str:
        num = self._rng.randint(1, 9999)
        street = self._rng.choice(_STREET_NAMES)
        city = self._rng.choice(_CITIES)
        state = self._rng.choice(_STATES)
        zipcode = self._rng.randint(10000, 99999)
        return f"{num} {street}, {city}, {state} {zipcode}"

    def id_number(self) -> str:
        prefix = self._rng.choice(["ID", "DL", "PP", "SSN"])
        digits = "".join(
            str(self._rng.randint(0, 9)) for _ in range(8)
        )
        return f"{prefix}{digits}"


# ---------------------------------------------------------------------------
# Text Templates by Domain
# ---------------------------------------------------------------------------

# Each template has {slot} placeholders that are filled with PII.
# The "domain" key is used for non-IID domain partitioning.
# The "pii_types" key lists which PII types appear in the template.

_TEMPLATES: List[Dict[str, Any]] = [
    # Medical
    {
        "domain": "medical",
        "pii_types": ["NAME", "PHONE", "SSN", "ADDRESS"],
        "template": (
            "Patient {NAME} (SSN: {SSN}) was admitted on Monday. "
            "Emergency contact phone: {PHONE}. "
            "Home address: {ADDRESS}. "
            "The attending physician noted stable vitals."
        ),
    },
    {
        "domain": "medical",
        "pii_types": ["NAME", "EMAIL", "PHONE"],
        "template": (
            "Dr. {NAME} can be reached at {EMAIL} or by phone at {PHONE}. "
            "Please schedule a follow-up appointment within two weeks."
        ),
    },
    # Financial
    {
        "domain": "financial",
        "pii_types": ["NAME", "CC", "EMAIL"],
        "template": (
            "Transaction alert for {NAME}: your card ending in {CC} "
            "was used for a purchase. Contact us at {EMAIL} if this "
            "was not authorized."
        ),
    },
    {
        "domain": "financial",
        "pii_types": ["NAME", "SSN", "ADDRESS"],
        "template": (
            "Loan application for {NAME}, SSN {SSN}. "
            "Mailing address: {ADDRESS}. "
            "Credit review is pending."
        ),
    },
    # Legal
    {
        "domain": "legal",
        "pii_types": ["NAME", "ID", "ADDRESS"],
        "template": (
            "Subpoena issued to {NAME}, identification number {ID}. "
            "Respondent address: {ADDRESS}. "
            "Failure to appear may result in contempt."
        ),
    },
    {
        "domain": "legal",
        "pii_types": ["NAME", "EMAIL", "PHONE"],
        "template": (
            "Counsel {NAME} filed a motion on behalf of the defendant. "
            "Contact: {EMAIL}, phone: {PHONE}."
        ),
    },
    # HR / Employment
    {
        "domain": "hr",
        "pii_types": ["NAME", "EMAIL", "SSN", "PHONE"],
        "template": (
            "New employee onboarding: {NAME}, email {EMAIL}, "
            "SSN {SSN}, phone {PHONE}. "
            "Please complete the I-9 form by end of week."
        ),
    },
    {
        "domain": "hr",
        "pii_types": ["NAME", "ADDRESS", "ID"],
        "template": (
            "Employee {NAME} has updated their address to {ADDRESS}. "
            "Employee ID: {ID}. HR records have been updated."
        ),
    },
    # Customer Service
    {
        "domain": "customer_service",
        "pii_types": ["NAME", "EMAIL", "CC"],
        "template": (
            "Dear {NAME}, your refund to card {CC} has been processed. "
            "A confirmation has been sent to {EMAIL}."
        ),
    },
    {
        "domain": "customer_service",
        "pii_types": ["NAME", "PHONE", "ADDRESS"],
        "template": (
            "Delivery for {NAME} scheduled to {ADDRESS}. "
            "We will call {PHONE} 30 minutes before arrival."
        ),
    },
    # Insurance
    {
        "domain": "insurance",
        "pii_types": ["NAME", "SSN", "PHONE", "ADDRESS"],
        "template": (
            "Claim filed by {NAME} (SSN: {SSN}). "
            "Contact number: {PHONE}. "
            "Property address: {ADDRESS}. "
            "An adjuster will be assigned within 48 hours."
        ),
    },
    # Education
    {
        "domain": "education",
        "pii_types": ["NAME", "EMAIL", "ID"],
        "template": (
            "Student {NAME} (ID: {ID}) has enrolled for the fall "
            "semester. Academic advisor contact: {EMAIL}."
        ),
    },
]

# PII type to generator method mapping
_PII_TYPE_TO_METHOD = {
    "NAME": "name",
    "EMAIL": "email",
    "PHONE": "phone",
    "SSN": "ssn",
    "CC": "credit_card",
    "ADDRESS": "address",
    "ID": "id_number",
}


# ---------------------------------------------------------------------------
# Synthetic PII Dataset
# ---------------------------------------------------------------------------

@dataclass
class SyntheticPIIDataset:
    """Generates synthetic text samples with embedded PII.

    Each sample is a dict containing:
      - ``"text"``: the raw text string with PII values filled in.
      - ``"input_ids"``: list of integer token IDs (character-level
        encoding or from a provided tokenizer).
      - ``"position_types"``: per-position type labels ("PUB", "SENS",
        "REG", etc.) for typed training.
      - ``"domain"``: the text domain (medical, financial, etc.).
      - ``"pii_spans"``: list of ``(start_char, end_char, pii_type)``
        tuples marking PII locations.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    seed : int
        Random seed for fully reproducible generation.
    tokenizer : Optional
        If provided, used for tokenization.  Must have an ``encode``
        method.  If ``None``, a simple character-level encoding is used.
    domains : Optional[List[str]]
        Restrict generation to templates from these domains.  If
        ``None``, all domains are used.
    pii_types : Optional[List[str]]
        Restrict to templates containing at least one of these PII
        types.  If ``None``, no filtering.
    max_length : int
        Maximum sequence length (in tokens).  Longer sequences are
        truncated.
    """

    num_samples: int = 100
    seed: int = 42
    tokenizer: Any = None
    domains: Optional[List[str]] = None
    pii_types: Optional[List[str]] = None
    max_length: int = 256

    def generate(self) -> List[Dict[str, Any]]:
        """Generate the dataset.

        Returns
        -------
        List[Dict[str, Any]]
            List of sample dicts.
        """
        rng = random.Random(self.seed)
        pii_gen = _PIIGenerator(rng)

        # Filter templates
        templates = list(_TEMPLATES)
        if self.domains is not None:
            domains_set = set(self.domains)
            templates = [t for t in templates if t["domain"] in domains_set]
        if self.pii_types is not None:
            pii_set = set(self.pii_types)
            templates = [
                t for t in templates
                if pii_set & set(t["pii_types"])
            ]

        if not templates:
            raise ValueError(
                "No templates match the specified domain/pii_types filters"
            )

        samples: List[Dict[str, Any]] = []

        for i in range(self.num_samples):
            template_info = rng.choice(templates)
            text, pii_spans = self._fill_template(
                template_info, pii_gen
            )

            # Tokenize
            if self.tokenizer is not None:
                input_ids = self.tokenizer.encode(
                    text, add_special_tokens=False
                )
            else:
                # Character-level encoding
                input_ids = [ord(c) for c in text]

            # Truncate
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]

            # Build per-position type labels
            position_types = self._build_position_types(
                text, pii_spans, len(input_ids)
            )

            samples.append({
                "text": text,
                "input_ids": input_ids,
                "position_types": position_types,
                "domain": template_info["domain"],
                "pii_spans": pii_spans,
                "attention_mask": [1] * len(input_ids),
            })

        return samples

    def _fill_template(
        self,
        template_info: Dict[str, Any],
        pii_gen: _PIIGenerator,
    ) -> Tuple[str, List[Tuple[int, int, str]]]:
        """Fill a template with generated PII values.

        Returns the filled text and a list of character-level PII spans.
        """
        template = template_info["template"]
        pii_spans: List[Tuple[int, int, str]] = []

        # Find all {SLOT} patterns and replace them
        result_parts: List[str] = []
        pos = 0
        current_offset = 0

        while pos < len(template):
            # Find next slot
            brace_start = template.find("{", pos)
            if brace_start == -1:
                result_parts.append(template[pos:])
                break

            brace_end = template.find("}", brace_start)
            if brace_end == -1:
                result_parts.append(template[pos:])
                break

            # Add text before the slot
            result_parts.append(template[pos:brace_start])
            current_offset += brace_start - pos

            # Extract slot name
            slot_name = template[brace_start + 1 : brace_end]

            # Generate PII value
            method_name = _PII_TYPE_TO_METHOD.get(slot_name)
            if method_name is not None:
                value = getattr(pii_gen, method_name)()
                pii_type = slot_name

                # Map some types to privacy classification
                if pii_type in ("SSN", "CC"):
                    type_label = "REG"
                else:
                    type_label = "SENS"

                span_start = current_offset
                span_end = current_offset + len(value)
                pii_spans.append((span_start, span_end, pii_type))

                result_parts.append(value)
                current_offset += len(value)
            else:
                # Unknown slot — keep as-is
                placeholder = template[brace_start : brace_end + 1]
                result_parts.append(placeholder)
                current_offset += len(placeholder)

            pos = brace_end + 1

        text = "".join(result_parts)
        return text, pii_spans

    def _build_position_types(
        self,
        text: str,
        pii_spans: List[Tuple[int, int, str]],
        num_tokens: int,
    ) -> List[str]:
        """Build per-token type labels from character-level PII spans.

        Uses character-level indexing (assuming character-level
        tokenization) or approximate mapping for subword tokenizers.
        """
        # Character-level type assignment
        char_types = ["PUB"] * len(text)
        for start, end, pii_type in pii_spans:
            label = "REG" if pii_type in ("SSN", "CC") else "SENS"
            for j in range(start, min(end, len(text))):
                char_types[j] = label

        # Map to token-level: for character-level tokenizers, 1:1
        # For subword tokenizers, we take the type of the first char
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            # Approximate: assign each token the type of its position
            # This is exact for char-level; approximate for subword
            position_types = []
            for i in range(num_tokens):
                if i < len(char_types):
                    position_types.append(char_types[i])
                else:
                    position_types.append("PUB")
            return position_types
        else:
            # Character-level: direct mapping
            return char_types[:num_tokens]


# ---------------------------------------------------------------------------
# Partitioning Functions
# ---------------------------------------------------------------------------

def partition_iid(
    dataset: List[Dict[str, Any]],
    num_clients: int,
    seed: int = 42,
) -> List[List[Dict[str, Any]]]:
    """Partition a dataset IID across clients.

    Each client receives a random, roughly equal subset of the data.
    This is the simplest partitioning and serves as a baseline.

    Parameters
    ----------
    dataset : List[Dict]
        The full dataset to partition.
    num_clients : int
        Number of clients.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    List[List[Dict]]
        One list of samples per client.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if not dataset:
        return [[] for _ in range(num_clients)]

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    # Distribute indices round-robin
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for i, idx in enumerate(indices):
        client_indices[i % num_clients].append(idx)

    return [
        [dataset[idx] for idx in c_indices]
        for c_indices in client_indices
    ]


def partition_non_iid_domain(
    dataset: List[Dict[str, Any]],
    num_clients: int,
    domain_skew: float = 0.8,
    seed: int = 42,
) -> List[List[Dict[str, Any]]]:
    """Partition a dataset with domain-based non-IID skew.

    Each client is assigned a "primary" domain and receives a
    disproportionate share of samples from that domain.

    Parameters
    ----------
    dataset : List[Dict]
        The full dataset.  Each sample must have a ``"domain"`` key.
    num_clients : int
        Number of clients.
    domain_skew : float
        Fraction of each client's data that comes from its primary
        domain.  0.0 = IID; 1.0 = fully non-IID (only primary domain).
        Values in (0, 1) blend primary and other domains.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    List[List[Dict]]
        One list of samples per client.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if not dataset:
        return [[] for _ in range(num_clients)]
    if not (0.0 <= domain_skew <= 1.0):
        raise ValueError("domain_skew must be in [0, 1]")

    rng = random.Random(seed)

    # Group samples by domain
    domain_pools: Dict[str, List[int]] = {}
    for idx, sample in enumerate(dataset):
        domain = sample.get("domain", "unknown")
        if domain not in domain_pools:
            domain_pools[domain] = []
        domain_pools[domain].append(idx)

    # Shuffle each pool
    for pool in domain_pools.values():
        rng.shuffle(pool)

    domains = sorted(domain_pools.keys())

    # Assign primary domains to clients (round-robin over domains)
    client_primary: List[str] = []
    for c in range(num_clients):
        client_primary.append(domains[c % len(domains)])

    # Target samples per client
    total = len(dataset)
    per_client = total // num_clients
    remainder = total % num_clients

    client_datasets: List[List[Dict[str, Any]]] = [
        [] for _ in range(num_clients)
    ]

    # Compute how many primary vs non-primary samples each client gets
    # Track remaining pool indices
    pool_pointers: Dict[str, int] = {d: 0 for d in domains}

    for c in range(num_clients):
        n_samples = per_client + (1 if c < remainder else 0)
        n_primary = int(n_samples * domain_skew)
        n_other = n_samples - n_primary

        primary = client_primary[c]

        # Draw from primary pool
        primary_pool = domain_pools[primary]
        ptr = pool_pointers[primary]
        drawn_primary_indices = primary_pool[ptr : ptr + n_primary]
        pool_pointers[primary] = ptr + len(drawn_primary_indices)

        # Draw from other pools (round-robin across non-primary domains)
        other_domains = [d for d in domains if d != primary]
        drawn_other_indices: List[int] = []
        if other_domains and n_other > 0:
            per_other = max(1, n_other // len(other_domains))
            for od in other_domains:
                od_pool = domain_pools[od]
                od_ptr = pool_pointers[od]
                take = min(per_other, len(od_pool) - od_ptr)
                drawn_other_indices.extend(od_pool[od_ptr : od_ptr + take])
                pool_pointers[od] = od_ptr + take
                if len(drawn_other_indices) >= n_other:
                    break
            drawn_other_indices = drawn_other_indices[:n_other]

        all_indices = drawn_primary_indices + drawn_other_indices
        rng.shuffle(all_indices)
        client_datasets[c] = [dataset[idx] for idx in all_indices]

    # Distribute any remaining unassigned samples
    assigned = set()
    for cd in client_datasets:
        for sample in cd:
            # Use id to track
            assigned.add(id(sample))

    unassigned: List[int] = []
    for idx in range(len(dataset)):
        # Check if this index was used
        found = False
        for d in domains:
            pool = domain_pools[d]
            if idx in pool[:pool_pointers.get(d, 0)]:
                found = True
                break
        # Simple approach: just check total counts
    # The above is imprecise; instead, count total assigned
    total_assigned = sum(len(cd) for cd in client_datasets)
    if total_assigned < total:
        # Collect remaining indices
        all_assigned_indices = set()
        for d in domains:
            pool = domain_pools[d]
            all_assigned_indices.update(pool[:pool_pointers[d]])
        remaining = [i for i in range(total) if i not in all_assigned_indices]
        rng.shuffle(remaining)
        for i, idx in enumerate(remaining):
            client_datasets[i % num_clients].append(dataset[idx])

    return client_datasets


def partition_non_iid_sensitivity(
    dataset: List[Dict[str, Any]],
    num_clients: int,
    seed: int = 42,
) -> List[List[Dict[str, Any]]]:
    """Partition a dataset with sensitivity-based non-IID skew.

    Each client specialises in different PII patterns.  Samples
    are grouped by the set of PII types they contain, and each
    client receives a disproportionate share of samples with
    specific PII type combinations.

    Parameters
    ----------
    dataset : List[Dict]
        The full dataset.  Each sample must have a ``"pii_spans"``
        key containing ``(start, end, pii_type)`` tuples.
    num_clients : int
        Number of clients.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    List[List[Dict]]
        One list of samples per client.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if not dataset:
        return [[] for _ in range(num_clients)]

    rng = random.Random(seed)

    # Group samples by their PII type signature
    type_groups: Dict[str, List[int]] = {}
    for idx, sample in enumerate(dataset):
        pii_spans = sample.get("pii_spans", [])
        pii_types = sorted(set(span[2] for span in pii_spans))
        signature = "|".join(pii_types) if pii_types else "NONE"
        if signature not in type_groups:
            type_groups[signature] = []
        type_groups[signature].append(idx)

    # Shuffle each group
    for group in type_groups.values():
        rng.shuffle(group)

    signatures = sorted(type_groups.keys())

    # Assign primary PII signatures to clients
    client_datasets: List[List[Dict[str, Any]]] = [
        [] for _ in range(num_clients)
    ]

    # Strategy: assign groups to clients round-robin, with overlap
    # Each group is primarily assigned to one client, but excess
    # samples spill to the next client.
    group_assignment: Dict[str, int] = {}
    for i, sig in enumerate(signatures):
        group_assignment[sig] = i % num_clients

    # Distribute: 70% of each group goes to the assigned client,
    # 30% is distributed to other clients
    SKEW = 0.7

    for sig, indices in type_groups.items():
        primary_client = group_assignment[sig]
        n_primary = int(len(indices) * SKEW)

        # Primary share
        primary_share = indices[:n_primary]
        remaining_share = indices[n_primary:]

        client_datasets[primary_client].extend(
            [dataset[idx] for idx in primary_share]
        )

        # Distribute remaining among other clients
        other_clients = [c for c in range(num_clients) if c != primary_client]
        if other_clients:
            for i, idx in enumerate(remaining_share):
                target = other_clients[i % len(other_clients)]
                client_datasets[target].append(dataset[idx])
        else:
            # Only one client
            client_datasets[primary_client].extend(
                [dataset[idx] for idx in remaining_share]
            )

    # Shuffle each client's dataset
    for cd in client_datasets:
        rng.shuffle(cd)

    return client_datasets
