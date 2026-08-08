import zipfile
import pickle
import io
import numpy as np

PT_PATH = "data/colehopf/burgers_colehopf.pt"
ARCHIVE = "burgers_colehopf"

DTYPE_MAP = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
}


def load(pt_path=PT_PATH, archive=ARCHIVE):
    z = zipfile.ZipFile(pt_path)

    class FakePersistentStorage:
        def __init__(self, storage_type_name, key, numel):
            raw = z.read(f"{archive}/data/{key}")
            dtype = DTYPE_MAP[storage_type_name]
            arr = np.frombuffer(raw, dtype=dtype, count=numel)
            self.arr = arr

    def find_class(module, name):
        if module == "torch" and name.endswith("Storage"):
            return name
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            def _rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks, *rest):
                arr = storage.arr
                n = int(np.prod(size)) if len(size) else 1
                sub = arr[storage_offset:storage_offset + n]
                out = np.reshape(sub, size)
                return out
            return _rebuild
        if module == "collections" and name == "OrderedDict":
            return dict
        raise RuntimeError(f"Unexpected global: {module} {name}")

    class Unpickler(pickle.Unpickler):
        pass
    Unpickler.find_class = staticmethod(find_class)

    def persistent_load(pid):
        assert pid[0] == "storage"
        storage_type_name = pid[1]
        key = pid[2]
        numel = pid[4]
        return FakePersistentStorage(storage_type_name, key, numel)

    data = z.read(f"{archive}/data.pkl")
    up = Unpickler(io.BytesIO(data))
    up.persistent_load = persistent_load
    result = up.load()
    return result


if __name__ == "__main__":
    d = load()
    for k, v in d.items():
        if hasattr(v, "shape"):
            print(k, v.shape, v.dtype)
        else:
            print(k, v)
