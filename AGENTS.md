1. use a good programming style:
1.1 no functions outside classes
1.2. group classes together into files logically
1.3. prevent circular imports
1.4 only the main file imports anything
1.5 prefer to inherit from existing classes whenever possible
1.6 prefer to override functions whenever possible

2. What you are NEVER allowed to do:
2.1 never implement any parts of code "simplified", "as stub", "as demo"
2.2 you are forbidden to simplify, cut short, shorten, stub, or restrict any part of existing code or algorithm

3. GOLDEN RULES
3.1 if a given task would be to extensive to do, then devide it into smaller managable steps and substebs that you present to the user as clickable buttons (task templates)
3.2 after every change to the code create / execute RELEVANT tests only
3.2.1 all tests must print RELEVANT data to the console
3.2.2 you must analyse these outputs for logical soundness, issues, errors warnings
3.2.3 you must fix all logical errors, issues, errors and warnings
3.2.4 if you can not currently fix a logical error, issue, error, warning then create / add to FAILEDTESTS.md
3.2.5 tests must NEVER monkeypatch anything, tests MUST use the production code only

4. YOUR WORKFLOW WHENEVER YOU ARE GIVEN A TASK, you must follow the sequence below exactly
4.1 check if FAILEDTESTS.md exists, if yes and there are tests not marked with [solved] try to solve as many of them as possible
4.1.1 mark any tests you solve as [solved]
4.2. think about the task you have been given, make a fully plan how to implement the task
4.2.1 figure out if the task is to extensive to be done in one run, if yes implement as much as possible of it and then devide it into smaller managable steps and substebs that you present to the user as clickable buttons (task templates)
4.2.2 test your changes
4.3. determine if your changes require creating / updating of requirements.txt, ensure that requirements.txt does not include any packages or package versions that are incompatible to each other. resolve any inconsistencies.

5. if during your work you can think up a rule to add to this AGENTS.md that would make sense, you may add it as long as it does not contradict or weaken any existing rule!
