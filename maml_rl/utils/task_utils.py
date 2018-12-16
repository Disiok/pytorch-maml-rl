def normalize_task_ids(task_distribution):
    """
    Normalize task ids.

    :param task_distribution [list<dict>]: A list of task configurations.
    :return                  [list<dict>]: A list of task configurations.
    """
    task_distribution = sorted(task_distribution, key=lambda t: t['task_id'])
    for task_id, task in enumerate(task_distribution):
        task['task_id'] = task_id
    return task_distribution
