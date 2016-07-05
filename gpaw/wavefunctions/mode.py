class Mode:
    def __init__(self, force_complex_dtype=False):
        self.force_complex_dtype = force_complex_dtype

    def write(self, writer):
        writer.write(mode=self.name)
        if self.force_complex_dtype:
            writer.write(force_complex_dtype=True)
