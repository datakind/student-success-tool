import typing as t
from datetime import date

from faker.providers import BaseProvider

from ...analysis.pdp import utils


class Provider(BaseProvider):
    def raw_course_record(self, normalize_col_names: bool = False) -> dict[str, object]:
        record = {
            "Student GUID": self.student_guid(),
            "Student Age": self.student_age(),
            "Race": self.race(),
            "Ethnicity": self.ethnicity(),
            "Gender": self.gender(),
            "Institution ID": self.institution_id(),
            "Cohort": self.cohort(),
            "Cohort Term": self.cohort_term(),
            "Academic Year": self.academic_year(),
            "Academic Term": self.academic_term(),
            "Course Prefix": self.course_prefix(),
            "Course Number": self.course_number(),
            "Section ID": self.section_id(),
            "Course Name": self.course_name(),
            "Course CIP": self.course_cip(),
            "Course Type": self.course_type(),
            "Math or English Gateway": self.math_or_english_gateway(),
            "Co-requisite Course": self.co_requisite_course(),
            "Course Begin Date": self.course_begin_date(),
            "Course End Date": self.course_end_date(),
            "Grade": self.grade(),
            "Number of Credits Attempted": self.number_of_credits_attempted(),
            "Number of Credits Earned": self.number_of_credits_earned(),
            "Delivery Method": self.delivery_method(),
            "Core Course": self.core_course(),
            "Core Course Type": self.core_course_type(),
            "Core Competency Completed": self.core_competency_completed(),
            "Enrolled at Other Institution(s)": self.enrolled_at_other_institution_s(),
            "Credential Engine Identifier": self.credential_engine_identifier(),
            "Course Instructor Employment Status": self.course_instructor_employment_status(),
            "Course Instructor Rank": self.course_instructor_rank(),
            "Enrollment Record at Other Institution(s) STATE(s)": self.enrollment_record_at_other_institution_s_state_s(),
            "Enrollment Record at Other Institution(s) CARNEGIE(s)": self.enrollment_record_at_other_institution_s_carnegie_s(),
            "Enrollment Record at Other Institution(s) LOCALE(s)": self.enrollment_record_at_other_institution_s_locale_s(),
        }
        if normalize_col_names:
            record = {
                utils.convert_to_snake_case(key): val for key, val in record.items()
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
        return f"{start_dt:%Y}-{end_dt:%Y}"

    def cohort_term(self) -> str:
        return self.random_element(["FALL", "WINTER", "SPRING", "SUMMER"])

    def academic_year(self, min_yr: int = 2010, max_yr: t.Optional[int] = None) -> str:
        start_dt = self.generator.date_between(start_date=date(min_yr, 1, 1))
        end_dt = start_dt.replace(year=start_dt.year + 1)
        return f"{start_dt:%Y}-{end_dt:%Y}"

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

    def _number_of_credits(self, min_value: float = 0.0) -> float:
        return self.generator.pyfloat(  # type: ignore
            min_value=min_value, max_value=20.0, right_digits=1
        )

    def number_of_credits_attempted(self) -> float:
        return self._number_of_credits(min_value=1.0)

    def number_of_credits_earned(self) -> float:
        return self._number_of_credits()

    def delivery_method(self) -> str:
        return self.random_element(["F", "O", "H"])

    def core_course(self) -> str:
        return self.random_element(["Y", "N"])

    def core_course_type(self) -> t.Optional[str]:
        # TODO
        return None

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

    def enrollment_record_at_other_institution_s_state_s(self) -> str:
        return self.generator.state_abbr(include_freely_associated_states=False)  # type: ignore

    def enrollment_record_at_other_institution_s_carnegie_s(self) -> str:
        return self.random_element(
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

    def enrollment_record_at_other_institution_s_locale_s(self) -> str:
        return self.random_element(["URBAN", "SUBURB", "TOWN/RURAL"])
