from os import path

def get_ref_model_path(args, model_class_name, dataset_class_name, suffix_str='base', model_setup=False):
    if args.exp == 'model_ref' or model_setup:
        outpath = path.join(args.experiment_path, '%s.HClass'%(model_class_name), '%s.dataset'%(dataset_class_name))
    else:
        outpath = path.join(args.experiment_path, "..", "model_ref", '%s.HClass'%(model_class_name), '%s.dataset'%(dataset_class_name))
    outpath = path.join(outpath, suffix_str)
    return outpath
