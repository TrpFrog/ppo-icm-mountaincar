import torch
from torch import Tensor


def get_device():
    # return 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cuda' if torch.cuda.is_available() else 'cpu'


def split_to_batches(*inputs_list, batch_size: int, dim=0):
    """
    Split tensor or list of tensors to batches
    :param inputs: Tensor or list of tensors, or dict of tensors
    :param batch_size:
    :param dim:
    :return:
    """

    def f(inputs):
        if isinstance(inputs, Tensor):
            return inputs.split(batch_size, dim=dim)

        HF_TRANSFORMERS_BATCH_ENCODING = "<class 'transformers.tokenization_utils_base.BatchEncoding'>"

        if isinstance(inputs, dict) or str(type(inputs)) == HF_TRANSFORMERS_BATCH_ENCODING:
            for k, v in inputs.items():
                inputs[k] = v.split(batch_size)

            n_set_of_batch = len(list(inputs.values())[0])
            ret = [{} for _ in range(n_set_of_batch)]
            for key, batched_tensor in inputs.items():
                for ret_dict, batch in zip(ret, batched_tensor):
                    ret_dict.update({key: batch})
            return ret

        is_namedtuple = isinstance(inputs, tuple) and hasattr(inputs, "_fields")
        if is_namedtuple:
            tmp = {}
            for k, v in inputs._asdict().items():
                tmp[k] = split_to_batches(v, batch_size=batch_size, dim=dim)

            ret = []
            for i in range(len(tmp[k])):
                ret.append(type(inputs)(*[v[i] for v in tmp.values()]))
            return ret

        if isinstance(inputs, tuple):
            tmp = []
            for x in inputs:
                tmp.append(split_to_batches(x, batch_size=batch_size, dim=dim))
            return list(zip(*tmp))

        else:
            raise TypeError(f"Unsupported type: {type(inputs)}")

    return [f(x) for x in inputs_list]
