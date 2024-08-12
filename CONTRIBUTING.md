# Contributing

## Code Guidelines

### Programming Language
The programming language of the project is Python 3.
The oldest currently supported version is Python 3.8.

### Programming Style
All source code shall be formatted according to the PEP 8 rules.
Additionally, the following rules apply:  
**Maximum Line Length**  
In code files, the maximum line length is 88 characters.
For Markdown files, there is no maximum line length, but there shall be at most one sentence per line.  
**Pylint**  
Newly added code should not introduce new issues picked up by pylint, except comments containing a TODO.

### Testing
For testing RKOpenMDAO uses pytest.
All new and changed code is expected to be at least covered by one system test, and additionally by as many unit tests as is useful.
Make sure to add tests for edge cases.

### Documentation
Every newly added function or class shall be documented.
If a file contains more then one class or function, it shall also be documented.

### Changing Existing Code
When existing code is changed, and it didn't fulfil the above points, any deviation from the guidelines has to be fixed as well.

## Development Workflow

The main branch shall always be of release quality.
Therefore, no direct push to main is permitted. 
As workflow, we use *feature branches* and *pull requests*.

Branches should have a descriptive name.
Ideally, they should prepend the kind of change that is introduced, e.g.:
- fix/that_one_issue instead of just that_one_issue
- feature/introduce_best_function instead of introduced_best_function  

Commits should be small, aiming for one change per commit.
Only working code should be pushed.

Before a change can be accepted into the main branch, a review has to take place.
This review shall make sure that the above guidelines are fulfilled, as well as ensure a reasonable technical quality of the code.
