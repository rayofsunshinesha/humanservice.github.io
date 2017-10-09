#+TITLE: Identity snap spell that canceled due to earning
#+SUBTITLE: Using SQL for doing Data Analysis
#+AUTHOR: Ruisha
#+EMAIL: ruishaz@gmail.com
#+STARTUP: showeverything
#+STARTUP: nohideblocks

* Before we start

** Case canceled due to earning 

- the self-sufficiency measure that human services current use to evaluate how well are spell recipents doing. 
- For example: If 5% of all cases that were canceled due to earning last quarter, 6% are canceled due to earning this quarter, self-sufficiency of spell recipent is increasing as the department believes.  

** Data tables

- case canceled due to earning are tagged in a table with chapen hall id and a date. 
- head of house hold spell information include caseid (primary key), chapen hall id, and spell start and end date.
- the goal is link above two and tag the spell with a 0/1 case canceled due to earning column.

* The food inspections data set

The data represents the inspections made in different facilities in
the area of Chicago.

There are different types of inspections, different types of
facilities and different results (or outcomes) of that
inspections. Also the data contains the
types of violations and text descriptions in free form about the
violations.

We will do this together using the functions  =partition by= 

#+BEGIN_SRC sql
//tag the specific spell that canceled due to earning for foodstamp
SQL:
DROP TABLE IF EXISTS c6.snap_cancelled_identify_hh_indcase_spells ;
CREATE TABLE c6.snap_cancelled_identify_hh_indcase_spells AS 
SELECT *, 1 as tag_cancel
FROM(
	SELECT cancel.ch_dpa_caseid, cancel.transaction_date,
	spell.recptno, spell.benefit_type, spell.end_date , 
	(cancel.transaction_date-spell.end_date) as days_between_end_cancel,
	FIRST_VALUE(spell.end_date) 
		OVER (PARTITION BY cancel.transaction_date 
		ORDER BY ABS(cancel.transaction_date-spell.end_date))
	FROM idhs.snap_cancelled_earnings_redacted cancel
	LEFT JOIN idhs.hh_indcase_spells spell
	ON cancel.ch_dpa_caseid=spell.ch_dpa_caseid
	WHERE (cancel.ch_dpa_caseid IS NOT NULL) AND 
	(spell.benefit_type LIKE '%foodstamp%')
) temp
WHERE ABS(days_between_end_cancel)<=60
ORDER BY ABS(days_between_end_cancel)
#+END_SRC

We would do the same thing for tanf spells 

#+BEGIN_SRC sql
DROP TABLE IF EXISTS c6.tanf_cancelled_identify_hh_indcase_spells ;
CREATE TABLE c6.tanf_cancelled_identify_hh_indcase_spells AS 
SELECT *, 1 as tag_cancel_tanf
FROM(
	SELECT cancel.ch_dpa_caseid, cancel.transaction_date,
	spell.recptno, spell.benefit_type, spell.end_date , 
	(cancel.transaction_date-spell.end_date) as days_between_end_cancel,
	FIRST_VALUE(spell.end_date) 
		OVER (PARTITION BY cancel.transaction_date 
		ORDER BY ABS(cancel.transaction_date-spell.end_date))
	FROM idhs.tanf_cancelled_earnings_redacted cancel
	LEFT JOIN idhs.hh_indcase_spells spell
	ON cancel.ch_dpa_caseid=spell.ch_dpa_caseid
	WHERE (cancel.ch_dpa_caseid IS NOT NULL) AND 
	(spell.benefit_type LIKE '%tanf%')
	order by cancel.ch_dpa_caseid,spell.end_date
) temp
WHERE ABS(days_between_end_cancel)<=60
ORDER BY ABS(days_between_end_cancel)
#+END_SRC

* Idea behind:

use all cases in idhs.snap_cancelled_earnings_redacted (where ch_dpa_caseid is not missing )
go merge with idhs.hh_indcase_spells spell to identify the specicfic spell that is canceled due to earning. 
tag the record of the smallest gap days btw transaction_date & spell.end_date 
if the gap is more than 60 days, I make the judgement that the spell is not related to the transaction_date.

The data represents the inspections made in different facilities in
the area of Chicago.

There are different types of inspections, different types of
facilities and different results (or outcomes) of that
inspections. Also the data contains the
types of violations and text descriptions in free form about the
violations.

Obviously, we have spatio-temporal data (i.e. the inspections happen
in a given time at some place).

* Some basic tasks in a data analysis project

- Cleaning the data
- Manipulating the data
- Create new /FeatureS/
- Create new views of the data
- Answering analytical questions
