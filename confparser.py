import os
import shlex as sh
from abc import abstractmethod, ABC
from configparser import ConfigParser
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Mapping, List, Dict, Tuple, Any, Optional

try:
    from ase.utils import search_current_git_hash

    GIT_HASH = search_current_git_hash('gpaw')
except ImportError:
    GIT_HASH = None


class ConfigParseError(Exception):
    """Raised upon error in the siteconfig configuration file."""


RawParams = Mapping[str, str]
ParsedParams = Mapping[str, Any]
ArgList = Tuple[str, ...]
PathList = Tuple[Path, ...]

def parse_bool(value: str) -> bool:
    if value.lower() in ("1", "on", "yes", "true", "y"):
        return True
    elif value.lower() in ("0", "off", "no", "false", "n"):
        return False
    raise ConfigParseError(f"Cannot transform '{value}' to a boolean (try 'true' or 'false' instead).")


def parse_opt_str(value: str) -> Optional[str]:
    if str:
        return value


def parse_flags(value: str) -> ArgList:
    do_not_split_token = "&&"
    do_not_split_space_token = "&~&"
    force_split_token = ":"
    flags = []
    pre_split = value.split(force_split_token)
    for sub_grp in pre_split:
        flags.extend(sh.split(sub_grp))
    flags = tuple(flag.replace(do_not_split_token, "") for flag in flags)
    flags = tuple(flag.replace(do_not_split_space_token, " ") for flag in flags)
    if (len(flags) == 1) and (flags[0] == ""):
        return ()
    return flags


def parse_paths(value: str) -> PathList:
    force_split_token = ":"
    return tuple(Path(p) for p in value.split(force_split_token) if p != '')


parser_map = {
    bool: parse_bool,
    ArgList: parse_flags,
    PathList: parse_paths,
    Optional[str]: parse_opt_str
}


@dataclass
class ConfigSection(ABC):

    @classmethod
    @abstractmethod
    def parse(cls, params: RawParams) -> "ConfigSection":
        ...


def expandvars(line: str) -> str:
    return os.path.expandvars(line)


@dataclass
class BaseConfigSection(ConfigSection):
    """Represents a configuration section.

    Known entries are represented by dataclass attributes. Unknown
    entries are put in the special `_extra_entries` attribute.
    """

    _extra_entries: ParsedParams = field(default_factory=dict)

    @classmethod
    def split_fields(cls, params: RawParams) -> Tuple[RawParams, RawParams]:
        my_fields = set(f.name for f in fields(cls) if f.name != "_extra_entries").intersection(set(params))
        extra_fields = set(params) - my_fields
        return {f: params[f] for f in my_fields}, {f: params[f] for f in extra_fields}

    @classmethod
    def parse_extra_fields(cls, extra_fields: RawParams) -> ParsedParams:
        fields = {k: expandvars(v) for k, v in extra_fields.items()}
        return fields

    @classmethod
    def parse_my_fields(cls, my_fields: RawParams) -> ParsedParams:
        params = {}
        for f in fields(cls):
            name = f.name
            if name not in my_fields:
                continue
            parse = parser_map.get(f.type, lambda x: x)
            params[name] = parse(expandvars(my_fields[name]))
        return params

    @classmethod
    def parse(cls, params: RawParams) -> ConfigSection:
        mine, extra = cls.split_fields(params)
        mine = cls.parse_my_fields(mine)
        extra = cls.parse_extra_fields(extra)
        return cls(**mine, _extra_entries=extra)

    def __str__(self):
        ret = []
        for f in fields(self.__class__):
            if f.name == "_extra_entries":
                continue
            val = getattr(self, f.name)
            ret.append(f"{f.name}: {val}")
        return "\n".join(ret)


@dataclass
class FixedBaseConfigSection(BaseConfigSection):
    """Represents a configuration section that fails to parse if any unknown entry is provided"""

    @classmethod
    def parse_extra_fields(cls, extra_fields: dict[str, str]):
        """Fail if unknown entries are present (e.g. typos)."""
        if extra_fields:
            raise ConfigParseError(f"Spurious entries: {list(extra_fields)!r}.")


@dataclass
class Library(FixedBaseConfigSection):
    enabled: bool = True
    libraries: ArgList = ()
    include_dirs: PathList = ()
    library_dirs: PathList = ()
    link_args: ArgList = ()
    compile_args: ArgList = ()
    add_to_rpath: bool = True


@dataclass
class Environment(FixedBaseConfigSection):
    platform_id: str = os.getenv("CPU_ARCH") or "unknown"
    c_compiler: str = os.getenv("CC") or "gcc"
    c_linker: str = os.getenv("LD") or os.getenv("CC") or "gcc"
    mpi_compiler: Optional[str] = os.getenv("MPICC")
    mpi_linker: Optional[str] = os.getenv("MPILD") or os.getenv("MPICC")
    parallel_interpreter: bool = False


MACRO_DEFAULTS = {
    "GPAW_ASYNC": "1",
    "GPAW_MPI2": "1",
    "NPY_NO_DEPRECATED_API": "7",
    "GPAW_NO_UNDERSCORE_CBLACS": "1",
    "GPAW_NO_UNDERSCORE_CSCALAPACK": "1",
    "NDEBUG": None,
}


class Macros(Dict[str, Optional[str]], ConfigSection):

    def __init__(self, **kwargs):
        super().__init__()
        self.update(MACRO_DEFAULTS)
        self.update(kwargs)

    def render(self) -> Tuple[List[Tuple[str, str]], List[str]]:
        defined, undefined = [], []
        for k, v in self.items():
            if v:
                defined.append((k, v))
            else:
                undefined.append(k)
        return defined, undefined

    @classmethod
    def parse(cls, params: RawParams) -> "Macros":
        return cls(**params)


@dataclass
class Setups(FixedBaseConfigSection):
    paths: PathList = ".:/usr/share/gpaw-setups:/usr/local/share/gpaw-setups"
    extra_paths: PathList = ""


@dataclass
class ExtraArguments(Library):
    compile_args: ArgList = ('-Wall', '-Wno-unknown-pragmas', '-std=c99')
    runtime_library_dirs: ArgList = ()


def retrieve_from_environment(name, field_name):
    """Retrieves information from the environment using name and field_name as keys"""
    env_var = f"GPAW_BUILD_{name}_{field_name}".upper()
    return os.getenv(env_var)


LIB_DEFAULTS = {
    "mpi": {},
    "blas": {"libraries": "blas:lapack"},
    "fftw": {"libraries": "fftw3_omp:fftw3"},
    "elpa": {"libraries": "elpa"},
    "scalapack": {"libraries": "scalapack"},
    "xc": {"libraries": "xc"},
    "vdwxc": {"libraries": "vdwxc"},
    "openmp": {
        "link_args": "-fopenmp -fopenmp-simd",
        "compile_args": "-fopenmp -fopenmp-simd",
    },
}


def gen_lib_defaults(name):
    _defaults = LIB_DEFAULTS[name]

    def _factory():
        return Library(**_defaults)

    return _factory


@dataclass
class Configuration:
    environment: Environment = field(default_factory=Environment)
    macros: Macros = field(default_factory=Macros)
    setups: Setups = field(default_factory=Setups)
    extra: ExtraArguments = field(default_factory=ExtraArguments)
    mpi: Library = field(default_factory=gen_lib_defaults("mpi"))
    blas: Library = field(default_factory=gen_lib_defaults("blas"))
    fftw: Library = field(default_factory=gen_lib_defaults("fftw"))
    elpa: Library = field(default_factory=gen_lib_defaults("elpa"))
    scalapack: Library = field(default_factory=gen_lib_defaults("scalapack"))
    xc: Library = field(default_factory=gen_lib_defaults("xc"))
    vdwxc: Library = field(default_factory=gen_lib_defaults("vdwxc"))
    openmp: Library = field(default_factory=gen_lib_defaults("openmp"))


@dataclass
class SiteConfigCompileEnvironment:
    libraries: List[str] = field(default_factory=list)
    library_dirs: List[str] = field(default_factory=list)
    include_dirs: List[str] = field(default_factory=list)
    runtime_library_dirs: List[str] = field(default_factory=list)
    define_macros: List[Tuple[str, str]] = field(default_factory=list)
    undef_macros: List[str] = field(default_factory=list)
    extra_link_args: List[str] = field(default_factory=list)
    extra_compile_args: List[str] = field(default_factory=list)
    extra_objects: List[str] = field(default_factory=list)
    environment: Mapping[str, str] = field(default_factory=dict)

    @property
    def dict(self):
        return asdict(self)


def parse_ini_config_files(*config_file_name: str) -> Configuration:
    """Parses GPAW build-time configuration in INI format"""
    parser = ConfigParser()
    parser.optionxform = str  # preserve case-sensitivity
    for filename in config_file_name:
        path = Path(filename).expanduser()
        if not path.is_file():
            continue
        parser.read(filename)
    parsed = {}
    for f in fields(Configuration):
        if f.name not in parser:
            continue
        parsed[f.name] = f.type.parse(parser[f.name])
    return Configuration(**parsed)


def add_macros_deps_from_config(config: Configuration) -> List[Tuple[str, str]]:
    extra_macros = []
    if config.mpi.enabled:
        extra_macros += [('PARALLEL', '1')]
    if not config.blas.enabled:
        extra_macros += [('GPAW_WITHOUT_BLAS', '1')]
    if not config.xc.enabled:
        extra_macros += [('GPAW_WITHOUT_LIBXC', '1')]
    if config.fftw.enabled:
        extra_macros += [('GPAW_WITH_FFTW', '1')]
    if config.scalapack.enabled:
        extra_macros += [('GPAW_WITH_SL', '1')]
    if config.vdwxc.enabled:
        extra_macros += [('GPAW_WITH_LIBVDWXC', '1')]
    if config.elpa.enabled:
        extra_macros += [('GPAW_WITH_ELPA', '1')]
    if GIT_HASH is not None:
        extra_macros += [('GPAW_GITHASH', GIT_HASH)]
    return extra_macros


def convert_configuration_to_compilation_env(config: Configuration) -> SiteConfigCompileEnvironment:
    env = SiteConfigCompileEnvironment()

    # Macros
    env.define_macros, env.undef_macros = config.macros.render()
    env.define_macros.extend(add_macros_deps_from_config(config))

    # Libraries
    for f in fields(Configuration):
        if not issubclass(f.type, Library):
            continue
        lib = getattr(config, f.name)
        if not lib.enabled:
            continue
        env.libraries.extend(lib.libraries)
        env.include_dirs.extend(lib.include_dirs)
        env.library_dirs.extend(lib.library_dirs)
        env.extra_link_args.extend(lib.link_args)
        env.extra_compile_args.extend(lib.compile_args)
        if lib.add_to_rpath:
            env.runtime_library_dirs.extend(lib.library_dirs)
    env.runtime_library_dirs.extend(config.extra.runtime_library_dirs)

    # Build and return env
    return env


if __name__ == "__main__":
    config = parse_ini_config_files("siteconfig.ini")
    print(config)
    env = convert_configuration_to_compilation_env(config)
    print(env.dict)

# serial_site_config = SiteConfig()
# mpi_site_config = SiteConfig()
