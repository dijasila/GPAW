def create_tasks():
    from myqueue.task import task
    return [
        task('surface.agts.py'),
        task('work_function.py', deps='surface.agts.py')]


if __name__ == '__main__':
    from pathlib import Path
    source = Path('Al100.py').read_text()
    source = source.replace('k = ...', 'k = 6')
    source = source.replace('N = ...', 'N = 5')
    exec(source)
