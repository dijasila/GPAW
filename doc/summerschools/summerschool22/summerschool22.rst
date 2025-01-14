.. _summerschool22:

=======================
CAMd Summer School 2022
=======================

Announcement:
https://www.fysik.dtu.dk/english/Research/CAMD/Events/CAMD-Summer-School-2022

.. highlight:: bash


Summer school exercises in Jupyter notebooks
============================================

The Summer School includes a number of :ref:`projects`, which are partly
formulated as Jupyter Notebooks.  In a Jupyter Notebook you are running the
calculations on the DTU central computing facilities, but the output is
displayed in your local browser.

Unfortunately, this requires some setup, which is described below.


The computer system
===================

The DTU computer system (known as the 'G-databar' for obscure reasons) consists of a login node (named ``login.gbar.dtu.dk`` or ``gbarlogin``) and a number of compute nodes.  Some of the compute nodes are reserved for batch jobs, some allow interactive jobs.  You will be running a Jupyter Notebook server on an interactive compute node, this server will run the Python jobs and will allow the browser on your laptop to see the output.  The latter unfortunately requires bypassing a firewall which would normally prevent you from accessing the compute nodes directly from the summer school site.


Instructions
============

The instructions depend on whether your laptop runs Windows, MacOS or Linux - the latter two are very similar in this context.


Windows users
-------------

.. toctree::
   :maxdepth: 1

   setupwin
   accesswin
   submitting


Mac and Linux users
-------------------

.. toctree::
   :maxdepth: 1

   setuplinmac
   accesslinmac
   submitting


iPad users (not recommended)
----------------------------

.. toctree::
   :maxdepth: 1

   ipad


Introductory slides
-------------------

The slides from the introduction to the exercises (Sunday) can be
found here:
:download:`Intro_projects_CAMD2022.pdf`


Projects
========

Choose a project according to your interests.  The projects contain
brief descriptions of what aspects of GPAW you learn from them.


.. toctree::
   :maxdepth: 3

   projects


Instructions for instructors
============================

Instructions for the people testing and developing the projects can be
found here:

.. toctree::

   instructors
