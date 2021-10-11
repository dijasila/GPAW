class OldStuff:
    def get_pseudo_wave_function(self, n):
        return self.calculation.ibz_wfs[0].wave_functions.data[n]

    def write(self, filename, mode=''):
        """Write calculator object to a file.

        Parameters
        ----------
        filename
            File to be written
        mode
            Write mode. Use ``mode='all'``
            to include wave functions in the file.
        """
        self.log(f'Writing to {filename} (mode={mode!r})\n')

        self.calculation.write(filename,
                               skip_wfs=mode != 'all')
