from django.contrib.postgres.fields import ArrayField
from django.db import models
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols

import multiprocessing 

class Molecule(models.Model):
    """
    Store server side keys
    """
    id = models.IntegerField(primary_key=True)
    smile = models.CharField(null=True, unique=True, max_length=100)
    fingerprint = ArrayField(models.IntegerField(), null=True, size=2048)
    gap = models.FloatField(null=True)

    @property
    def molecule(self):
        return Chem.MolFromSmiles(self.smile)

    def calculate_fingerprint(self):
        fingerprint = FingerprintMols.FingerprintMol(self.molecule)
        return [2 * idx + val for idx, val in enumerate(fingerprint)]
    
    def save(self, *args, **kwargs):
        if self.smile:
            self.fingerprint = list(self.calculate_fingerprint())

        return super(Molecule, self).save(*args, **kwargs)

    @classmethod
    def process_mol(cls, val_tuple):
        idx, smile = val_tuple
        m, _ = Molecule.objects.get_or_create(id=idx)
        m.smile = smile
        # m.gap = gap
        m.save()

    @classmethod
    def load_molecules(cls, csv_file, num_cores, start_index, testing=True):
        """
        Reads in the data from a pandas csv
        """
        print ("LOADING DATAFRAME")

        df = pd.read_csv(csv_file)

        print ("DATAFRAME LOADED")

        # id all the tuples
        smile_gap_tuples = enumerate(df.smiles.values)

        # don't process tuples we've already seen
        smile_gap_tuples = list(smile_gap_tuples)[start_index:]

        # if we're resting, just run the first 10
        if testing:
            smile_gap_tuples = smile_gap_tuples[:10]

        p = multiprocessing.Pool(num_cores)

        p.map(cls.process_mol, smile_gap_tuples)
