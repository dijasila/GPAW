from gpaw import mpi
import time

class WESTDummy:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.count = 0
        
    def main_loop(self):
        self.count = 0

        while True:
            if self.server_has_failed():
                break
            if not self.server_is_done():
                time.sleep(3)
                continue
            self.count += 1
            mpi.world.barrier()
            time.sleep(10)

            if mpi.rank == 0:
                print("WEST Dummy ran")

                
            self.remove_server_lock()
            if self.count > 3:
                self.send_done_signal()
                break
            mpi.world_barrier()

    def server_is_done(self):
        import os
        fname = "./" + self.input_file.split(".")[0] + ".lock"
        serverdone = os.path.exists(fname)
        return serverdone

    def remove_server_lock(self):
        import os
        if mpi.rank != 0:
            return
        fname = "./" + self.input_file.split(".")[0] + ".lock"
        if os.path.exists(fname):
            os.remove(fname)

    def send_done_signal(self):
        if mpi.rank == 0:
            fname = "./" + self.input_file.split(".")[0] + ".DONE"
            with open(fname, "w+") as f:
                f.write("DONE")

    def server_has_failed(self):
        import os
        fname = self.input_file.split(".")[0] + ".FAILED"
        return os.path.exists(fname)
            

if __name__ == "__main__":
    import sys
    input_file, output_file = sys.argv[1:3]
    westclient = WESTDummy(input_file, output_file)
    westclient.main_loop()
