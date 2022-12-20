task_to_keys = {
    "eurlex57k" : ("text", None),
    "eurlex4k" : ("text", None),
    "amazon13k" : ("text", None),
    "wiki1m" : ("text", None),
}

task_to_label_keys = {
    "eurlex57k" : 'label',
    "eurlex4k" : 'label',
    "amazon13k": 'label',
    "wiki1m" : "label"
}



dataset_classification_type = {
    "eurlex57k" : 'multi_label_classification',
    "eurlex4k"  : 'multi_label_classification',
    "amazon13k"  : 'multi_label_classification',
    "wiki1m"  : 'multi_label_classification',
}

dataset_to_numlabels = {
    "eurlex57k" : 4271,
    "eurlex4k"  : 3956,
    "amazon13k"  : 13330,
    "wiki1m"  : 1200000, # TODO: Enter precise value, though doesn't matter
}