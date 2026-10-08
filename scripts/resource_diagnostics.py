"""Read-only Linux cgroup evidence for a child killed while training.

A SIGKILL alone cannot identify OOM. Only an increase in the cgroup's OOM-kill
counter provides kernel evidence that a process in this group was OOM-killed.
"""
from pathlib import Path


def _text(path):
    try:
        return path.read_text(encoding='ascii').strip()
    except (OSError, UnicodeError):
        return None


def _number(path):
    value = _text(path)
    try:
        return int(value) if value is not None and value != 'max' else None
    except ValueError:
        return None


def _counters(path):
    result = {}
    for line in (_text(path) or '').splitlines():
        fields = line.split()
        if len(fields) == 2:
            try:
                result[fields[0]] = int(fields[1])
            except ValueError:
                pass
    return result


def memory_snapshot(root=Path('/sys/fs/cgroup'), membership=Path('/proc/self/cgroup')):
    """Support namespaced Docker roots and ordinary Linux v1/v2 memberships."""
    root, membership = Path(root), Path(membership)
    candidates = [root, root / 'memory']
    for line in (_text(membership) or '').splitlines():
        fields = line.split(':', 2)
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        # Do not follow unexpected parent traversal in kernel/path fixtures.
        path = Path(relative.lstrip('/'))
        if '..' in path.parts:
            continue
        if controllers == '':
            candidates.insert(0, root / path)
        elif 'memory' in controllers.split(','):
            candidates.insert(0, root / 'memory' / path)
    for directory in candidates:
        events = _counters(directory / 'memory.events')
        if events:
            return {'version': 2, 'current_bytes': _number(directory / 'memory.current'),
                    'peak_bytes': _number(directory / 'memory.peak'),
                    'limit_bytes': _number(directory / 'memory.max'), 'events': events}
        events = _counters(directory / 'memory.oom_control')
        if events:
            return {'version': 1, 'current_bytes': _number(directory / 'memory.usage_in_bytes'),
                    'peak_bytes': _number(directory / 'memory.max_usage_in_bytes'),
                    'limit_bytes': _number(directory / 'memory.limit_in_bytes'),
                    'allocation_failures': _number(directory / 'memory.failcnt'), 'events': events}
    return {'version': None, 'available': False}


def exit_evidence(returncode, before, after):
    same_group = before.get('version') is not None and before.get('version') == after.get('version')
    prior, current = before.get('events', {}), after.get('events', {})
    delta = {key: max(0, value - prior[key]) for key, value in current.items()
             if same_group and key in prior}
    oom_killed = delta.get('oom_kill', 0) > 0 or delta.get('oom_group_kill', 0) > 0
    return {'exit_code': returncode,
            'classification': 'cgroup_oom_kill' if oom_killed else 'child_failed_cause_unconfirmed',
            'oom_kill_confirmed_in_cgroup': oom_killed, 'memory_event_delta': delta,
            'before': before, 'after': after,
            'note': 'Cgroup OOM counters apply to this container/group; SIGKILL alone does not prove an OOM.'}
