"""Developer quiz."""
from __future__ import annotations
from dataclasses import dataclass
from random import shuffle
from textwrap import wrap


@dataclass
class Question:
    text: str
    choises: list[str] | None = None
    solutions: list[str] | None = None
    shuffle: bool = True
    extra_check: bool = False

    def ask(self) -> bool:
        """Ask question and check result."""
        print()
        text = self.text.rstrip('?') + '?'
        for line in text.splitlines():
            print('\n'.join(wrap(line)))
        if self.choises:
            N = len(self.choises)
            indices = list(range(N))
            if self.shuffle:
                shuffle(indices)
            i0 = len(self.text) - len(text)
            n0 = indices.index(i0) + 1
            for n, i in enumerate(indices, start=1):
                print(f'{n}: {self.choises[i]}')
            answer = input(f'[1-{N}]: ')
            if answer != str(n0):
                return False
            if self.extra_check:
                return Question(text='Really?', choises=['yes', 'no']).ask()
            return True

        answer = input()
        return self.solutions is None or answer in self.solutions


dev = 'https://wiki.fysik.dtu.dk/gpaw/devel'

questions = [
    Question(text='What, ... is your name?'),
    Question(text='What, ... is your quest?',
             choises=['To become a GPAW-developer']),
    Question(text='What, ... is your favourite programming language?',
             choises=['Python', 'Other'],
             shuffle=False),
    Question('What, ... is the air-speed of unladen swallow???',
             choises=['10 m/s',
                      '20 miles an hour',
                      'depends on the kind of swallow (African or European)']),
    Question(text='Here is some important developer information:\n\n'
             f'  {dev}/workflow.html\n'
             f'  {dev}/testing.html\n'
             f'  {dev}/writing_documentation.html\n\n'
             'Did you read all of these three pages??',
             choises=['no', 'yes'],
             extra_check=True),
    Question(text='What framework does GPAW use for its test suite??',
             choises=['home-made system',
                      'pytest',
                      'unittest']),
    Question(text='What is the name of the environment variable that the '
             '"gpw_files" fixture uses for the folder to store gpw-files in?',
             solutions=['GPW_TEST_FILES', '$GPW_TEST_FILES'])]


def main() -> None:
    """Ask questions."""
    print('To become a gpaw developer, you must answer these '
          f'{len(questions)} questions ...')
    shuffle(questions)
    for question in questions:
        ok = question.ask()
        if ok:
            print('\033[92mCorrect!\033[0m')
        else:
            print('\033[91mAuuuuuuuugh!\033[0m')
            return
    print('\nRight. Off you go!')


if __name__ == '__main__':
    main()
