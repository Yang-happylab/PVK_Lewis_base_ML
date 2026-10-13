"""Finite center/arm templates from the supplied molecular design diagram."""

from functools import lru_cache
from itertools import combinations_with_replacement
import random

from rdkit import Chem

from structure_features import canonical_smiles


# Port indices refer to these explicit molecular graphs, not canonical atom order.
RINGS = {
    'Ph': ('c1ccccc1', (0, 2)),
    'Py': ('n1ccccc1', (1, 5)),
    'Fn': ('o1cccc1', (1, 4)),
    'Th': ('s1cccc1', (1, 4)),
    'Cp': ('C1=CC=CC1', (0, 3)),
}
ARM_NAMES = ['H', 'C', 'N', 'O', 'S', 'Cl', 'Cp', 'Fn', 'Th', 'Ph', 'Py']
CENTER_TEMPLATES = {}


def _center(name, smiles, ports, ring):
    CENTER_TEMPLATES[name] = {
        'mol': Chem.MolFromSmiles(smiles), 'ports': tuple(ports),
        'arity': len(ports), 'ring': ring,
    }


for element, arities in [('C', (1, 2, 3)), ('N', (1, 2, 3)),
                         ('O', (1, 2)), ('S', (1, 2)), ('Cl', (1,))]:
    for arity in arities:
        _center(f'{element}_{arity}', element, [0] * arity, False)
for name in ('Ph', 'Py'):
    smiles, ports = RINGS[name]
    _center(f'{name}_1', smiles, ports[:1], True)
    _center(f'{name}_2', smiles, ports, True)
for name in ('Fn', 'Th', 'Cp'):
    _center(f'{name}_2', *RINGS[name], True)
_center('Ph_3', 'c1ccccc1', (0, 2, 4), True)
_center('Tz_3', 'n1cncnc1', (1, 3, 5), True)
_center('Ph_4', 'c1ccccc1', (0, 1, 3, 4), True)
_center('Pz_4', 'n1ccncc1', (1, 2, 4, 5), True)

# Each two-ring center has one specified outer port on each ring.
for left, right in [('Py', 'Py'), ('Ph', 'Ph'), ('Py', 'Ph'),
                    *combinations_with_replacement(('Fn', 'Th', 'Cp'), 2)]:
    lsmiles, lp = RINGS[left]
    rsmiles, rp = RINGS[right]
    lm, rm = Chem.MolFromSmiles(lsmiles), Chem.MolFromSmiles(rsmiles)
    offset = lm.GetNumAtoms()
    rw = Chem.RWMol(Chem.CombineMols(lm, rm))
    rw.AddBond(lp[0], offset + rp[0], Chem.BondType.SINGLE)
    Chem.SanitizeMol(rw)
    CENTER_TEMPLATES[f'{left}_{right}_2'] = {
        'mol': rw.GetMol(), 'ports': (lp[1], offset + rp[1]),
        'arity': 2, 'ring': True,
    }


def is_valid(sample):
    if len(sample) != 5 or sample[0] not in CENTER_TEMPLATES:
        return False
    arity = CENTER_TEMPLATES[sample[0]]['arity']
    return all(a in ARM_NAMES for a in sample[1:arity + 1]) and all(
        a == 'None' for a in sample[arity + 1:]
    )


@lru_cache(maxsize=50000)
def _assemble(sample):
    if not is_valid(sample):
        raise ValueError(f'Invalid center/arm combination: {sample}')
    center = CENTER_TEMPLATES[sample[0]]
    rw = Chem.RWMol(center['mol'])
    for port, arm in zip(center['ports'], sample[1:]):
        if arm == 'H':
            continue  # Leave this port unsubstituted; RDKit supplies implicit H.
        arm_ring = arm in RINGS
        if arm_ring:
            smiles, ports = RINGS[arm]
            fragment, endpoint = Chem.MolFromSmiles(smiles), ports[0]
        else:
            fragment, endpoint = Chem.MolFromSmiles(arm), 0
        # Atomic-to-atomic: two carbons; ring-to-atomic: one; ring-to-ring: direct.
        spacer_length = 2 - int(center['ring']) - int(arm_ring)
        previous = port
        for _ in range(spacer_length):
            carbon = rw.AddAtom(Chem.Atom('C'))
            rw.AddBond(previous, carbon, Chem.BondType.SINGLE)
            previous = carbon
        offset = rw.GetNumAtoms()
        rw.InsertMol(fragment)
        rw.AddBond(previous, offset + endpoint, Chem.BondType.SINGLE)
    Chem.SanitizeMol(rw)
    return canonical_smiles(Chem.MolToSmiles(rw))


def assemble(sample):
    return _assemble(tuple(sample))


def generate_valid_sample():
    center = random.choice(list(CENTER_TEMPLATES))
    arity = CENTER_TEMPLATES[center]['arity']
    return [center] + random.choices(ARM_NAMES, k=arity) + ['None'] * (4 - arity)
