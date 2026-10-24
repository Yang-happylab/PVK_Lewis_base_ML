"""The 32 two-dimensional structure descriptors defined for this library."""

from itertools import combinations

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors


FEATURE_COLUMNS = [
    'n_N_primary', 'n_N_secondary', 'n_N_tertiary', 'n_N_pyridine',
    'n_N_triazine', 'n_N_pyrazine', 'n_O_hydroxyl', 'n_O_ether',
    'n_O_furan', 'n_S_thiol', 'n_S_thioether', 'n_S_thiophene',
    'n_Cl_donor', 'n_ring_benzene', 'n_ring_CPD', 'n_C_sp3',
    'n_C_branch', 'n_rotatable', 'n_ring_link', 'n_N_Ar', 'n_O_Ar',
    'n_S_Ar', 'T_NN', 'T_NO', 'T_NS', 'T_NCl', 'T_OO', 'T_OS',
    'T_OCl', 'T_SS', 'T_SCl', 'T_ClCl',
]
EXTRACTION_SETTINGS = {
    'rdkit_version': rdBase.rdkitVersion,
    'rotatable_bonds': 'Strict',
    'distance_use_bond_order': False,
    'pair_aggregation': 'mean_inverse_distance_v2',
    'stereochemistry': 'removed for two-dimensional identity',
}
HALOGEN_PROXY_SETTINGS = {
    'source_elements': ['Cl', 'Br', 'I'],
    'model_feature_element': 'Cl',
    'scope': 'terminal single-bonded carbon halogens in posterior structures',
}


def molecule(smiles):
    """Parse one neutral connected molecule, preserving its covalent chemistry."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f'Invalid or disconnected structure: {smiles}')
    if any(a.GetFormalCharge() or a.GetAtomicNum() not in (1, 6, 7, 8, 16, 17)
           for a in mol.GetAtoms()):
        raise ValueError(f'Structure outside the neutral C/N/O/S/Cl rules: {smiles}')
    mol = Chem.RemoveHs(mol)
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def canonical_smiles(smiles):
    return Chem.MolToSmiles(molecule(smiles), canonical=True)


def halogen_proxy_smiles(smiles):
    """Keep the input identity separate from a Cl-proxy structure for old models."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f'Invalid or disconnected structure: {smiles}')
    allowed = {1, 6, 7, 8, 16, 17, 35, 53}
    if any(atom.GetFormalCharge() or atom.GetAtomicNum() not in allowed
           for atom in mol.GetAtoms()):
        raise ValueError(f'Structure outside the neutral C/N/O/S/Cl/Br/I rules: {smiles}')
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ('Br', 'I'):
            continue
        bonds = list(atom.GetBonds())
        if (len(bonds) != 1 or bonds[0].GetBondType() != Chem.BondType.SINGLE
                or atom.GetNeighbors()[0].GetSymbol() != 'C'):
            raise ValueError(f'Unsupported Br/I donor environment: {smiles}')
        atom.SetAtomicNum(17)
    Chem.SanitizeMol(mol)
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True)


def describe_structure(smiles, *, halogen_proxy=False):
    """Return ordered 32 raw values; candidate donors are not actual denticity."""
    if halogen_proxy:
        smiles = halogen_proxy_smiles(smiles)
    mol = molecule(smiles)
    values = dict.fromkeys(FEATURE_COLUMNS, 0)
    rings = [frozenset(r) for r in mol.GetRingInfo().AtomRings()]
    environments = {}
    for ring in set(rings):
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        elements = [a.GetSymbol() for a in atoms]
        aromatic = all(a.GetIsAromatic() for a in atoms)
        if (len(ring) in (5, 6)
                and all(a.GetHybridization() == Chem.HybridizationType.SP3 for a in atoms)
                and all(sum(i in other for other in rings) == 1 for i in ring)):
            continue
        if len(ring) == 6 and aromatic and set(elements) == {'C'}:
            values['n_ring_benzene'] += 1
        elif len(ring) == 5 and set(elements) == {'C'} and not aromatic:
            bonds = [b for b in mol.GetBonds()
                     if b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring]
            if sum(b.GetBondType() == Chem.BondType.DOUBLE for b in bonds) != 2:
                raise ValueError(f'Unsupported carbon five-membered ring: {smiles}')
            values['n_ring_CPD'] += 1
        elif len(ring) == 6 and aromatic and set(elements) == {'C', 'N'}:
            nitrogens = [a.GetIdx() for a in atoms if a.GetSymbol() == 'N']
            kind = {1: 'pyridine', 2: 'pyrazine', 3: 'triazine'}.get(len(nitrogens))
            ring_distances = [len(Chem.GetShortestPath(mol, i, j)) - 1
                              for i, j in combinations(nitrogens, 2)]
            if kind is None or (kind == 'pyrazine' and ring_distances != [3]) or (
                kind == 'triazine' and any(d != 2 for d in ring_distances)
            ):
                raise ValueError(f'Unsupported aromatic N ring: {smiles}')
            environments.update({i: 'n_N_' + kind for i in nitrogens})
        elif len(ring) == 5 and aromatic and (
            elements.count('O') == 1 or elements.count('S') == 1
        ) and elements.count('C') == 4:
            for atom in atoms:
                if atom.GetSymbol() in ('O', 'S'):
                    environments[atom.GetIdx()] = (
                        'n_O_furan' if atom.GetSymbol() == 'O' else 'n_S_thiophene'
                    )
        else:
            raise ValueError(f'Unsupported ring: {smiles}')

    donors = []
    for atom in mol.GetAtoms():
        symbol, idx = atom.GetSymbol(), atom.GetIdx()
        if symbol == 'C':
            if atom.GetHybridization() == Chem.HybridizationType.SP3:
                values['n_C_sp3'] += 1
                values['n_C_branch'] += int(not atom.IsInRing() and atom.GetDegree() >= 3)
            continue
        if symbol == 'H':
            continue
        category = environments.get(idx)
        if category is None:
            neighbors = list(atom.GetNeighbors())
            single_carbon = all(a.GetSymbol() == 'C' for a in neighbors) and all(
                b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds()
            )
            degree, hydrogens = atom.GetDegree(), atom.GetTotalNumHs()
            if single_carbon and not atom.GetIsAromatic():
                if symbol == 'N' and degree in (1, 2, 3) and hydrogens == 3 - degree:
                    category = ('n_N_primary', 'n_N_secondary', 'n_N_tertiary')[degree - 1]
                elif symbol in ('O', 'S') and degree in (1, 2) and hydrogens == 2 - degree:
                    category = {'O': ('n_O_hydroxyl', 'n_O_ether'),
                                'S': ('n_S_thiol', 'n_S_thioether')}[symbol][degree - 1]
                elif symbol == 'Cl' and degree == 1:
                    category = 'n_Cl_donor'
        if category is None:
            raise ValueError(f'Unclassified donor {idx} ({symbol}): {smiles}')
        values[category] += 1
        donors.append((idx, symbol))
        if symbol in ('N', 'O', 'S') and not atom.GetIsAromatic():
            values[f'n_{symbol}_Ar'] += sum(
                a.GetSymbol() == 'C' and a.GetIsAromatic() for a in atom.GetNeighbors()
            )

    values['n_rotatable'] = rdMolDescriptors.CalcNumRotatableBonds(
        mol, rdMolDescriptors.NumRotatableBondsOptions.Strict,
    )
    values['n_ring_link'] = sum(
        b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()
        and b.GetBeginAtom().IsInRing() and b.GetEndAtom().IsInRing()
        for b in mol.GetBonds()
    )
    distances = Chem.GetDistanceMatrix(mol, useBO=False)
    order = {'N': 0, 'O': 1, 'S': 2, 'Cl': 3}
    pair_counts = dict.fromkeys(FEATURE_COLUMNS[22:], 0)
    for (i, a), (j, b) in combinations(donors, 2):
        pair = ''.join(sorted((a, b), key=order.get))
        values['T_' + pair] += 1.0 / distances[i, j]
        pair_counts['T_' + pair] += 1
    for name, count in pair_counts.items():
        if count:
            values[name] /= count
    return np.array([values[name] for name in FEATURE_COLUMNS], dtype=float)
