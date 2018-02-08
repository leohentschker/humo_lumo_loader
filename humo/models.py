from django.contrib.postgres.fields import ArrayField
from django.contrib.postgres.indexes import GinIndex
from django.db import models
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from deepchem.feat.coulomb_matrices import CoulombMatrixEig

import multiprocessing

class Molecule(models.Model):
    """
    Store server side keys
    """
    id = models.IntegerField(primary_key=True)
    smile = models.CharField(null=True, unique=True, max_length=100)
    fingerprint = ArrayField(models.IntegerField(), null=True, size=2048)
    coulomb_eigenspace = ArrayField(models.FloatField(), null=True)
    adjacency_eigenspace = ArrayField(models.FloatField(), null=True)
    gap = models.FloatField(null=True)

    class Meta:
        indexes = [GinIndex(fields=["fingerprint"])]

    coulomb_featureizer = CoulombMatrixEig(remove_hydrogens=True, max_atoms=200)

    @property
    def molecule(self):
        return Chem.MolFromSmiles(self.smile)

    def calculate_fingerprint(self):
        fingerprint = FingerprintMols.FingerprintMol(self.molecule)
        return [2 * idx + val for idx, val in enumerate(fingerprint)]

    def calculate_coulomb(self):
        mol = Chem.AddHs(self.molecule)
        Chem.AllChem.EmbedMultipleConfs(mol,1)
        Chem.AllChem.UFFOptimizeMoleculeConfs(mol)
        matrix = self.coulomb_featureizer.coulomb_matrix(mol)
        eigenvalues, _ = np.linalg.eig(matrix)
        return [float(v) for v in np.sort(eigenvalues[0].real)[::-1]]

    def calculate_adjacency(self):
        matrix = Chem.rdmolops.GetAdjacencyMatrix(self.molecule)
        eigenvalues, _ = np.linalg.eig(matrix)
        return [float(v) for v in np.sort(eigenvalues[0].real)[::-1]]
    
    def save(self, *args, **kwargs):
        # if self.smile:
        #     self.fingerprint = list(self.calculate_fingerprint())
        # if self.smile:
        #     self.coulomb_eigenspace = self.calculate_coulomb()
        if self.smile:
            self.adjacency_eigenspace = self.calculate_adjacency()

        return super(Molecule, self).save(*args, **kwargs)

    @classmethod
    def process_mol(cls, val_tuple):
        idx, (smile, gap) = val_tuple
        m, _ = Molecule.objects.get_or_create(id=idx)
        m.smile = smile
        m.gap = gap
        print (m.id)
        m.save()

    @classmethod
    def to_csv(cls, file):
        """
        Converts the class method to CSV
        """
        molecules = Molecule.objects.filter(id__gt=500000)[:2]
        print ("LOADED QUERY")
        vals = molecules.values_list("smile", "coulomb_eigenspace", "gap")
        data = []
        for smile, cle, gap in vals:
            row = {
                "smile": smile,
                "gap": gap,
            }

            for idx, v in enumerate(cle):
                row[idx] = v

            data.append(row)

        print (data)
        df = pd.DataFrame(data)
        df.to_csv(file)

    @classmethod
    def load_molecules(cls, csv_file, num_cores, start_index, end_index, testing=True):
        """
        Reads in the data from a pandas csv
        """
        print ("LOADING DATAFRAME")

        df = pd.read_csv(csv_file)

        print ("DATAFRAME LOADED")

        # id all the tuples
        smile_gap_tuples = enumerate(zip(df.smiles.values, df.gap.values))

        # don't process tuples we've already seen
        smile_gap_tuples = list(smile_gap_tuples)[start_index:end_index]

        # if we're resting, just run the first 10
        if testing:
            smile_gap_tuples = smile_gap_tuples[:10]

        p = multiprocessing.Pool(num_cores)

        p.map(cls.process_mol, smile_gap_tuples)