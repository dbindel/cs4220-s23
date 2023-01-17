---
title:      Syllabus
layout:     main
---

> The purpose of computing is insight, not numbers.  -- Richard Hamming

### Lecture and section information

CS 4220/CS 5223/MATH 4260, Spring 2023  
Lecture time: MWF 11:20-12:10  
Lecture location: Gates G01

### Staff and office hours

Prof: [David Bindel](http://www.cs.cornell.edu/~bindel)  
425 Gates Hall  
Phone: 607-255-5395  
Email: bindel@cornell.edu  
OH: TBA (or [by appointment](https://outlook.office365.com/owa/calendar/BindelOH@cornellprod.onmicrosoft.com/bookings/))

TA: Eric Chen  
Email: emc348@cornell.edu  
OH: TBA

TA: Kangbo Li  
Email: kl935@cornell.edu  
OH: TBA

TA: Abhay Singh  
Email: as2626@cornell.edu  
OH: TBA

### Catalog description

4 credits.  Introduction to the fundamentals of numerical
linear algebra: direct and iterative methods for linear systems,
eigenvalue problems, singular value decomposition. In the second half
of the course, the above are used to build iterative methods for
nonlinear systems and for multivariate optimization. Strong emphasis
is placed on understanding the advantages, disadvantages, and limits
of applicability for all the covered techniques. Computer programming
is required to test the theoretical concepts throughout the
course.

[cs4210]: http://www.cs.cornell.edu/Courses/CS4210/2014fa/

### Prerequisites

Linear algebra at the level of MATH 2210 or 2940
or equivalent and a CS 1 course in any language.  We will assume you
remember your calculus and can pick up Julia.  Recommended but not
essential: either CS 3220 or one additional mathematics course 
numbered 3000 or above.
This course can be taken before or after CS 4210/MATH 4250.

## Texts

### Recommended text

- Ascher and Greif, [_A First Course in Numerical Methods_][ag] ([etext][age])

We will cover roughly chapters 1-9.

### References

Some linear algebra references:

- Meyer, [_Matrix Analysis and Applied Linear Algebra_][meyer]
- Lay, [_Linear Algebra and its Applications_][lay]
- Strang, [_Linear Algebra and its Applications_][strang]
- Strang's excellent online course is [here][strangocw]

The [Julia language home page](http://julialang.org/) has many pointers
to using the Julia programming language; see in particular the
[learning](http://julialang.org/learning/) link at the top of the
home page.

[ag]: http://bookstore.siam.org/cs07/
[age]: http://epubs.siam.org/doi/book/10.1137/9780898719987

[meyer]: http://www.amazon.com/gp/product/0898714540/qid=1137779618/sr=2-1/ref=pd_bbs_b_2_1/002-5247186-8320001
[lay]: http://www.amazon.com/Linear-Algebra-Its-Applications-Edition/dp/0321385179
[strang]: http://www.amazon.com/gp/product/0155510053/qid=1137779745/sr=2-1/ref=pd_bbs_b_2_1/002-5247186-8320001
[strangocw]: http://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/

[ncm]: http://www.mathworks.com/moler/index_ncm.html
[itc]: http://www.ec-securehost.com/SIAM/OT117.html
[pratap]: http://www.amazon.com/gp/product/0195179374/qid=1137779327/sr=8-1/ref=pd_bbs_1/002-5247186-8320001

## Approximate lecture schedule

- Week 1: Matrix manipulations; review of vector spaces, norms, and
  singular values.
- Week 2: Sensitivity and conditioning.  Floating point.
- Week 3: Gaussian elimination
- Week 4: Conditioning, error estimation, and refinement of linear
  systems.  Methods for sparse, banded, and other structured linear
  systems.
- Week 5: Introduction to least squares and basic methods.
- Week 6: Sensitivity, ill-posedness, and regularization in least
  squares.
- Week 7: Eigenvalue problem theory, applications, and basic
  iterations.
- Week 8: Nonlinear equations and optimization in 1D.
- Week 9: Stationary iterations and Krylov iterations for linear
  systems.
- Week 10: Nonlinear equations, optimization, and multi-dimensional
  Newton.
- Week 11: Modified Newton, quasi-Newton, and gradient methods.
- Week 12: Globalization with line search, trust regions, and
  continuation.
- Week 13: Theory and survey of methods for constrained optimization.
- Week 14: Methods for structured optimization problems.  Derivative
  free methods.
- Week 15: Review.

## Course work

### Readings and the problem du jour

Readings from the course text (or notes) will be listed on the course
page before class.  *You are responsible for reading before lecture*.

For most lectures, there will be a "problem of the day" related to
the class material.  These problems are not graded, but you should
try them (and try to understand them) as a way to better learn the
material -- and as a way to study for the exams.

### Class work

Prior to each lecture, I will pose 1-2 questions (posted online) that
you should be able to answer by the end.  You should submit answers to
these questions, along with any notes on things that you found
particularly confusing (or interesting!).  Each submission counts for
a third of a point on the final grade (for a total of 10% of the
grade, equal to one project).

It is also critical for us to have your feedback about how the class
is going, both to improve the class for the current semester and to
make the class better for future semesters.  We will solicit
non-anonymous comments around the midterm, and at the end of the
semester will check with the college to see who has completed course
evaluation surveys (though we obviously cannot check to see whether
your feedback is useful!).  Participating in these feedback activities
counts toward your grade via points on the midterm and final.

### Homework and projects

There will be six one-week homeworks, assigned Friday and due the
following Friday.  These problems will involve a mix of short answers,
plots, and computations done in Julia. Homework should be
typed and submitted on CMS.  After they are graded,
homework scores will be posted to CMS.  Regrade requests must be
submitted within one week of receiving the graded homework.

There will be three two-week programming projects, to be done either
alone or in a group of two.  Projects will involve solving a larger
problem, and should be done in Julia.  For projects,
you will need to submit both codes and a writeup PDF file on CMS.

In order to provide timely, high-quality feedback, we may *not* always
grade all problems in a homework or pieces to a project.  Instead, we
will focus our grading efforts on providing feedback on a few key
points.  We will provide written solutions so that you can evaluate
yourself for problems where we do not grade in detail.

### Paper project

Students taking CS 5223 will be required to read a relevant paper and
write a short set of notes on the topic as well as at least two
homework-style problems (and their solutions).  A proposal for the
project should be submitted before the midterm; the project itself may
be submitted up to the last day of classes.

### Exams

There will be one midterm and one final exam:

 - Midterm: March 3-10
 - Final Exam: TBD, depending on central scheduling

Both exams are take home.  A couple points on the final will go
toward course evaluation.

### Grading

Your final grade in CS 4220 will be computed from grades on the
assignments and exams using the following weights:

 - Class work: 10%
 - Homework: 6% times 5 homeworks (best of 6)
 - Projects: 10% times 3 projects
 - Midterm: 15%
 - Final: 15%

If you are taking the course as CS 5223, the weights will be:

 - Class work: 10%
 - Homework: 5% times 5 homeworks (best of 6)
 - Paper project: 5%
 - Projects: 10% times 3 projects
 - Midterm: 15%
 - Final: 15%

## Learning outcomes

At the end of the semester, you will be able to:

- Analyze sources of error in numerical algorithms and reason about
  problem stability and how it influences the accuracy of numerical
  computations.
- Choose appropriate numerical algorithms to solve linear algebra
  problems (linear systems, least squares problems, and eigenvalue
  problems), taking into account problem structure.
- Formulate nonlinear equations and constrained and unconstrained 
  optimization problem for solution on a computer.
- Analyze the local convergence of nonlinear solver algorithms.
- Reason about global convergence of nonlinear solver algorithms.
- Use numerical methods to solve problems of practical interest from
  data science, machine learning, and engineering.

## Course policies

### Inclusivity and accommodation

We (Cornell as a whole, CS as a department, and I as the course
instructor) are commited to full inclusion in education for everyone.
Services and reasonable accommodations are available, whether you are
facing permanent or temporary disabilities, immigration status issues,
mental health or other personal challenges, or other types of learning
challenges.  If circumstances affect your ability to participate, let
me know.  Some resources that might be of use include:

 - [Office of Student Disability Services](https://sds.cornell.edu)
 - [Cornell Health Counseling and Psychological
   Services](https://health.cornell.edu/services/counseling-psychiatry)
 - [Undocumented/DACA Student
   Support](https://dos.cornell.edu/undocumented-daca-support/undergraduate-admissions-financial-aid)

### Ingredients for success

To be successful in the course, I ask that you focus on

1. Prepare.  Course notes will be posted ahead of class, and we will
   provide pointers to supplementary reading.  Come armed
   with questions!
2. Engage.  In class, we will ask you to ask questions and answer
   questions.  This is part of your grade, and a chance for you to
   practice what you've learned (and help us understand when you're confused).
2. Start homework early.  We want to help you figure out the homework,
   but to manage this we need time for you to get confused, ask us for help,
   and repeat a few times.

### Late work policy

Unless otherwise stated, all work is due by 11:59 pm on the due date.
All homework and projects should be submitted via the course
management system (CMS); you are encouraged to submit early versions,
since resubmissions up to the deadline are counted without penalty.
For each assignment, up to three "slip days" are allowed.  Over the
semester, you may use a total of six slip days.  You may not use slip
days for the take-home midterm.

If you need additional accommodation, ask in writing in advance, with
rationale and a plan for when you will be able to submit the work.

### Collaboration

An assignment is an academic document, like a journal article.
When you turn it in, you are claiming everything in it is your
original work, *unless you cite a source for it*.

You are welcome to discuss homework and projects among yourselves in
general terms.  However, you should not look at code or writeups from
other students, or allow other students to see your code or writeup,
even if the general solution was worked out together.  Unless we
explicitly allow it on an assignment, we will not credit code or
writeups that are shared between students (or teams, in the case of
projects).

If you get an idea from a classmate, the TA, a book or other published
source, or elsewhere, please provide an appropriate citation.  This is
not only critical to maintaining academic integrity, but it is also an
important way for you to give credit to those who have helped you out.
When in doubt, cite!  Code or writeups with appropriate citations will
never be considered a violation of academic integrity in this class
(though you will not receive credit for code or writeups that were
shared when you should have done them yourself).

### Academic Integrity

We expect academic integrity from everyone.  School is stressful,
and you may feel pressure from your coursework or other factors,
but that is no reason for dishonesty!  If you feel you can't complete
the work on the own, come talk to the professor, the TA, or your advisor,
and we can help you figure out what to do.

For more information, see Cornell's
[Code of Academic Integrity](http://cuinfo.cornell.edu/Academic/AIC.html).

### Emergency procedures

In the event of a major campus emergency, course requirements, deadlines, and
grading percentages are subject to changes that may be necessitated by a
revised semester calendar or other circumstances.  Any such announcements will
be posted to [the course home page](index.html).
