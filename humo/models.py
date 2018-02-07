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
        return FingerprintMols.FingerprintMol(self.molecule)
    
    def save(self, *args, **kwargs):
        if not self.fingerprint and self.smile:
            self.fingerprint = list(self.calculate_fingerprint())

        return super(Molecule, self).save(*args, **kwargs)

    @classmethod
    def load_molecules(cls, csv_file, num_cores, start_index, testing=True):
        """
        Reads in the data from a pandas csv
        """
        df = pd.read_csv(csv_file)

        # id all the tuples
        smile_gap_tuples = enumerate(zip(df.smiles.values, df.gap.values))

        # don't process tuples we've already seen
        smile_gap_tuples = list(smile_gap_tuples)[start_index:]

        # if we're resting, just run the first 10
        if testing:
            smile_gap_tuples = smile_gap_tuples[:10]

        p = multiprocessing.Pool(num_cores)

        def process_mol(val_tuple):
            idx, (smile, gap) = val_tuple
            print (idx, smile, gap)
            return
            m, _ = Molecule.objects.get_or_create(id=idx)
            m.smile = smile
            m.gap = gap
            m.save()
            return Chem.MolFromSmiles(smile)

        p.map(process_mol, smile_gap_tuples)