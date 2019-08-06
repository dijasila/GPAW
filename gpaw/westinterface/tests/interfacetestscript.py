from gpaw.westinterface import WESTInterface

calc_file = "subcalc.gpw"


wi = WESTInterface(calc_file, computer="gbar", use_dummywest=True)


opts = {}
opts["Input"] = "submitin.xml"
opts["Output"] = "submitout.xml"
opts["GPAWNodes"] = 2
opts["WESTNodes"] = 2
opts["Time"] = "00:05:00"

wi.run(opts)
