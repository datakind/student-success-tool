import typing as t
from datetime import date

from faker.providers import BaseProvider

from ... import utils


class Provider(BaseProvider):
    def raw_course_record(
        self, cohort_record: t.Optional[dict] = None, normalize_col_names: bool = False
    ) -> dict[str, object]:
        # use existing values where records overlap
        if cohort_record is not None:
            cr = cohort_record  # for more compact lines below
            student_guid = cr.get("student_guid", cr["Student GUID"])
            student_age = cr.get("student_age", cr["Student Age"])
            race = cr.get("race", cr["Race"])
            ethnicity = cr.get("ethnicity", cr["Ethnicity"])
            gender = cr.get("gender", cr["Gender"])
            institution_id = cr.get("institution_id", cr["Institution ID"])
            cohort = cr.get("cohort", cr["Cohort"])
            cohort_term = cr.get("cohort_term", cr["Cohort Term"])
            _has_enrollment_other_inst: bool = (
                cr.get(
                    "most_recent_bachelors_at_other_institution_state",
                    cr["Most Recent Bachelors at Other Institution STATE"],
                )
                is not None
            )
        else:
            student_guid = self.student_guid()
            student_age = self.student_age()
            race = self.race()
            ethnicity = self.ethnicity()
            gender = self.gender()
            institution_id = self.institution_id()
            cohort = self.cohort()
            cohort_term = self.cohort_term()
            _has_enrollment_other_inst: bool = self.generator.random.random() < 0.25  # type: ignore
        # derive a few more values, for self-consistency
        _min_academic_yr = int(cohort.split("-")[0])
        academic_year = self.academic_year(min_yr=_min_academic_yr)
        _min_course_yr = int(academic_year.split("-")[0])
        _max_course_yr = _min_course_yr + 1
        course_begin_date = self.course_begin_date(
            min_yr=_min_course_yr, max_yr=_max_course_yr
        )
        course_end_date = self.course_end_date(
            min_yr=_min_course_yr, max_yr=_max_course_yr
        )
        course_begin_date, course_end_date = sorted(
            [course_begin_date, course_end_date], reverse=False
        )  # ensure that begin date comes before end date
        core_course = self.core_course()
        core_course_type = self.core_course_type(core_course)
        num_credits_attempted = self.number_of_credits_attempted()
        num_credits_earned = self.number_of_credits_earned(
            max_value=num_credits_attempted
        )
        record = {
            "Student GUID": student_guid,
            "Student Age": student_age,
            "Race": race,
            "Ethnicity": ethnicity,
            "Gender": gender,
            "Institution ID": institution_id,
            "Cohort": cohort,
            "Cohort Term": cohort_term,
            "Academic Year": academic_year,
            "Academic Term": self.academic_term(),
            "Course Prefix": self.course_prefix(),
            "Course Number": self.course_number(),
            "Section ID": self.section_id(),
            "Course Name": self.course_name(),
            "Course CIP": self.course_cip(),
            "Course Type": self.course_type(),
            "Math or English Gateway": self.math_or_english_gateway(),
            "Co-requisite Course": self.co_requisite_course(),
            "Course Begin Date": course_begin_date,
            "Course End Date": course_end_date,
            "Grade": self.grade(),
            "Number of Credits Attempted": num_credits_attempted,
            "Number of Credits Earned": num_credits_earned,
            "Delivery Method": self.delivery_method(),
            "Core Course": core_course,
            "Core Course Type": core_course_type,
            "Core Competency Completed": self.core_competency_completed(),
            "Enrolled at Other Institution(s)": self.enrolled_at_other_institution_s(),
            "Credential Engine Identifier": self.credential_engine_identifier(),
            "Course Instructor Employment Status": self.course_instructor_employment_status(),
            "Course Instructor Rank": self.course_instructor_rank(),
            "Enrollment Record at Other Institution(s) STATE(s)": self.enrollment_record_at_other_institution_s_state_s(
                _has_enrollment_other_inst
            ),
            "Enrollment Record at Other Institution(s) CARNEGIE(s)": self.enrollment_record_at_other_institution_s_carnegie_s(
                _has_enrollment_other_inst
            ),
            "Enrollment Record at Other Institution(s) LOCALE(s)": self.enrollment_record_at_other_institution_s_locale_s(
                _has_enrollment_other_inst
            ),
        }
        if normalize_col_names:
            record = {
                utils.misc.convert_to_snake_case(key): val
                for key, val in record.items()
            }
        return record

    def student_guid(self) -> str:
        return self.numerify("#####!")  # type: ignore

    def institution_id(self) -> str:
        return self.numerify("#####!")  # type: ignore

    def student_age(self) -> str:
        return self.random_element(["20 AND YOUNGER", ">20 - 24", "OLDER THAN 24"])

    def race(self) -> str:
        return self.random_element(
            [
                "NONRESIDENT ALIEN",
                "AMERICAN INDIAN OR ALASKA NATIVE",
                "ASIAN",
                "BLACK OR AFRICAN AMERICAN",
                "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
                "WHITE",
                "HISPANIC",
                "TWO OR MORE RACES",
                "UNKNOWN",
            ]
        )

    def ethnicity(self) -> str:
        return self.random_element(["H", "N", "UK"])

    def gender(self) -> str:
        return self.random_element(["M", "F", "P", "X", "UK"])

    def cohort(self, min_yr: int = 2010, max_yr: t.Optional[int] = None) -> str:
        start_dt = self.generator.date_between(start_date=date(min_yr, 1, 1))
        end_dt = start_dt.replace(year=start_dt.year + 1)
        return f"{start_dt:%Y}-{end_dt:%y}"

    def cohort_term(self) -> str:
        return self.random_element(["FALL", "WINTER", "SPRING", "SUMMER"])

    def academic_year(self, min_yr: int = 2010, max_yr: t.Optional[int] = None) -> str:
        start_dt = self.generator.date_between(
            start_date=date(min_yr, 1, 1),
            end_date=(date(max_yr, 1, 1) if max_yr else "today"),
        ).replace(day=1)
        end_dt = start_dt.replace(year=start_dt.year + 1)
        return f"{start_dt:%Y}-{end_dt:%y}"

    def academic_term(self) -> str:
        return self.random_element(["FALL", "WINTER", "SPRING", "SUMMER"])

    # TODO: more realistic course prefix and number?

    def course_prefix(self) -> str:
        return self.lexify("????").upper()

    def course_number(self) -> str:
        return self.numerify("##!")

    def section_id(self) -> str:
        return self.numerify("##!.#")

    def course_name(self) -> str:
        return " ".join(self.generator.words(nb=3, part_of_speech="noun")).upper()

    def course_cip(self) -> str:
        # TODO: make this six-digit CIP code more realistic
        return self.numerify("##.####")

    def course_type(self) -> str:
        return self.random_element(
            ["CU", "CG", "CC", "CD", "EL", "AB", "GE", "NC", "O"]
        )

    def math_or_english_gateway(self) -> str:
        return self.random_element(["M", "E", "NA"])

    def co_requisite_course(self) -> str:
        return self.random_element(["Y", "N"])

    def _course_date(self, min_yr: int = 2010, max_yr: t.Optional[int] = None) -> date:
        _end_date = date(max_yr, 1, 1) if max_yr is not None else "today"
        return self.generator.date_between(  # type: ignore
            start_date=date(min_yr, 1, 1), end_date=_end_date
        )

    def course_begin_date(
        self, min_yr: int = 2010, max_yr: t.Optional[int] = None
    ) -> date:
        return self._course_date(min_yr, max_yr)

    def course_end_date(
        self, min_yr: int = 2010, max_yr: t.Optional[int] = None
    ) -> date:
        return self._course_date(min_yr, max_yr)

    def grade(self) -> str:
        # TODO: use weighting for more realistic distribution?
        # find out if grades can ever be given as continuous values
        return self.random_element(
            ["0", "1", "2", "3", "4", "P", "F", "I", "W", "A", "M", "O"]
        )

    def _number_of_credits(
        self, min_value: float = 0.0, max_value: float = 20.0
    ) -> float:
        return self.generator.pyfloat(  # type: ignore
            min_value=min_value,
            max_value=max(max_value, min_value + 1e-3),
            right_digits=1,
        )

    def number_of_credits_attempted(self) -> float:
        return self._number_of_credits(min_value=1.0)

    def number_of_credits_earned(self, max_value: float = 20.0) -> float:
        return self._number_of_credits(max_value=max_value)

    def delivery_method(self) -> str:
        return self.random_element(["F", "O", "H"])

    def core_course(self) -> str:
        return self.random_element(["Y", "N"])

    def core_course_type(self, core_course: t.Optional[str] = None) -> t.Optional[str]:
        if core_course and core_course == "N":
            return None
        else:
            # TODO
            return "CORE COURSE TYPE"

    def core_competency_completed(self) -> str:
        return self.random_element(["Y", "N"])

    def enrolled_at_other_institution_s(self) -> str:
        return self.random_element(["Y", "N"])

    def credential_engine_identifier(self) -> t.Optional[str]:
        return None

    def course_instructor_employment_status(self) -> str:
        return self.random_element(["PT", "FT"])

    def course_instructor_rank(self) -> str:
        return self.random_element(["1", "2", "3", "4", "5", "6", "7"])

    def enrollment_record_at_other_institution_s_state_s(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return (
            self.generator.state_abbr(include_freely_associated_states=False)
            if has_enrollment
            else None
        )  # type: ignore

    def enrollment_record_at_other_institution_s_carnegie_s(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return (
            self.random_element(
                [
                    "High traditional",
                    "Mixed traditional/nontraditional",
                    "High nontraditional",
                    "Associate's Dominant",
                    "Mixed Baccalaureate/Associate's Colleges",
                    "Arts & sciences focus",
                    "Diverse fields",
                    "M1: Master's Colleges and Universities - Larger programs",
                    "M2: Master's Colleges and Universities – Medium programs",
                    "M3: Master's Colleges and Universities – Small programs",
                    "R1: Doctoral universities – very high research activity",
                    "R2: Doctoral universities – high research activity",
                    "D/PU: Doctoral/professional universities",
                ]
            )
            if has_enrollment
            else None
        )

    def enrollment_record_at_other_institution_s_locale_s(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return (
            self.random_element(["URBAN", "SUBURB", "TOWN/RURAL"])
            if has_enrollment
            else None
        )
