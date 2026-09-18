import random
import os
import numpy as np
from numpy.random import permutation
from omegaconf import OmegaConf, ListConfig
from hydra.utils import instantiate


def map_attack_clients(clients_attack_types, prop_attack_clients, num_of_clients):
    """Map each client rank to its attack type."""
    if isinstance(clients_attack_types, ListConfig):
        clients_attack_types = list(clients_attack_types)
    if isinstance(prop_attack_clients, ListConfig):
        prop_attack_clients = list(prop_attack_clients)

    if isinstance(clients_attack_types, str):
        assert isinstance(
            prop_attack_clients, float
        ), f"If clients_attack_types is str, prop_attack_clients must be a float."
        clients_attack_types = [clients_attack_types]
        prop_attack_clients = [prop_attack_clients]
    else:
        assert isinstance(
            prop_attack_clients, list
        ), f"If clients_attack_types is a list, prop_attack_clients must be a list too."
        assert len(clients_attack_types) == len(
            prop_attack_clients
        ), "The length of clients_attack_types and prop_attack_clients must match."

    clients_map = {i: "no_attack" for i in range(num_of_clients)}
    for prob, attack_type in zip(prop_attack_clients, clients_attack_types):
        num_attacks = int(
            prob * num_of_clients
        )
        available_clients = [
            client_id
            for client_id, type_ in clients_map.items()
            if type_ == "no_attack"
        ]
        selected_clients = random.sample(
            available_clients, num_attacks
        )
        for client_id in selected_clients:
            print(f"Client {client_id} will attack with {attack_type}")
            clients_map[client_id] = (
                attack_type
            )
    return clients_map


def set_attack_rounds(rounds_prop, num_rounds, attack_scheme):
    """Sample which rounds clients attack according to rounds_prop."""
    assert attack_scheme in [
        "no_attack",
        "constant",
        "random_rounds",
        "random_clients",
        "random_rounds_random_clients",
    ], f"Attack scheme must be one of ['no_attack', 'constant', 'random_rounds', 'random_clients', 'random_rounds_random_clients'], you provide: {attack_scheme}"
    if attack_scheme != "no_attack":
        assert (
            rounds_prop != 0.0
        ), f"You set attack scheme {attack_scheme}, but proportion of attack rounds is zero"
    if attack_scheme == "constant" and rounds_prop != 1.0:
        print(
            "You set constant attack scheme, but your proportion of attacked rounds is not equal to 1. Set it manualy"
        )
        rounds_prop = 1.0
    num_attacks_rounds = int(rounds_prop * num_rounds)
    attack_rounds = random.sample([i for i in range(num_rounds)], num_attacks_rounds)
    if "random_rounds" in attack_scheme:
        print(f"List of attacked rounds: {sorted(attack_rounds)}")
    if 0 not in attack_rounds:
        attack_rounds.append(0)
    return attack_rounds


def permute_client_map(client_map):
    """Permute ranks of attacked clients."""
    permute_ranks = permutation(np.array(list(client_map.keys())))
    permuted_client_map = {}
    for i, rank in enumerate(permute_ranks):
        permuted_client_map[i] = client_map[rank]
        if client_map[rank] != "no_attack":
            print(f"Client {i} will attack with {client_map[rank]}")
    print()
    return permuted_client_map


def set_client_map_round(client_map, attacked_rounds, attack_scheme, cur_round):
    """Return the client map for the current round according to the scheme."""
    if cur_round not in attacked_rounds:
        return {rank: "no_attack" for rank in client_map.keys()}
    elif "random_clients" in attack_scheme:
        return permute_client_map(client_map)
    else:
        return client_map


def load_attack_configs(cfg, attack_types):
    """Load attack configs, overriding params from cfg."""
    attack_configs = {"no_attack": None}
    if isinstance(attack_types, str):
        attack_types = [attack_types]
    else:
        attack_types = list(attack_types)

    for attack_type in attack_types:
        if attack_type == "no_attack":
            continue
        script_dir = os.path.dirname(os.path.abspath(__file__))
        attack_cfg_path = os.path.join(
            script_dir, "../", f"configs/attacks/{attack_type}.yaml"
        )
        attack_config = OmegaConf.load(attack_cfg_path)
        for key in attack_config.keys():
            attack_config[key] = cfg.federated_params.get(key, attack_config[key])

        attack_configs[attack_type] = attack_config
    return attack_configs


def add_attack_functionality(client_instance, attack_type, attack_config):
    """Instantiate attack client and apply it to the given client."""
    attack_client = instantiate(attack_config)
    attack_client_instance = attack_client.apply_attack(client_instance)
    return attack_client_instance
