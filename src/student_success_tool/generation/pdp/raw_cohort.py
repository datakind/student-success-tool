import typing as t
from datetime import date

from faker.providers import BaseProvider

from ... import utils


class Provider(BaseProvider):
    def raw_cohort_record(
        self,
        min_cohort_yr: int = 2010,
        max_cohort_yr: t.Optional[int] = None,
        normalize_col_names: bool = False,
        institution_id: t.Optional[str] = None,
        student_guid: t.Optional[str] = None,
    ) -> dict[str, object]:
        # some fields are inputs to others; compute them first, accordingly
        enrollment_type = self.enrollment_type()
        enrollment_intensity_first_term = self.enrollment_intensity_first_term()
        number_of_credits_attempted_year_1 = self.number_of_credits_attempted_year_1()
        number_of_credits_attempted_year_2 = self.number_of_credits_attempted_year_2()
        number_of_credits_attempted_year_3 = self.number_of_credits_attempted_year_3()
        number_of_credits_attempted_year_4 = self.number_of_credits_attempted_year_4()
        _has_enrollment_other_inst: bool = self.generator.random.random() < 0.25

        # TODO: handle other cases, e.g. gateway course attempted/completed/grades
        record = {
            "Student GUID": student_guid
            if student_guid is not None
            else self.student_guid(),
            "Institution ID": institution_id
            if institution_id is not None
            else self.institution_id(),
            "Cohort": self.cohort(min_yr=min_cohort_yr, max_yr=max_cohort_yr),
            "Cohort Term": self.cohort_term(),
            "Student Age": self.student_age(),
            "Enrollment Type": enrollment_type,
            "Enrollment Intensity First Term": enrollment_intensity_first_term,
            "Math Placement": self.math_placement(),
            "English Placement": self.english_placement(),
            "Dual and Summer Enrollment": self.dual_and_summer_enrollment(),
            "Race": self.race(),
            "Ethnicity": self.ethnicity(),
            "Gender": self.gender(),
            "First Gen": self.first_gen(),
            "Pell Status First Year": self.pell_status_first_year(),
            "Attendance Status Term 1": self.attendance_status_term_1(
                enrollment_type, enrollment_intensity_first_term
            ),
            "Credential Type Sought Year 1": self.credential_type_sought_year_1(),
            "Program of Study Term 1": self.program_of_study_term_1(),
            "Program of Study Year 1": self.program_of_study_year_1(),
            "GPA Group Term 1": self.gpa_group_term_1(),
            "GPA Group Year 1": self.gpa_group_year_1(),
            "Number of Credits Attempted Year 1": number_of_credits_attempted_year_1,
            "Number of Credits Earned Year 1": self.number_of_credits_earned_year_1(
                max_value=number_of_credits_attempted_year_1
            ),
            "Number of Credits Attempted Year 2": number_of_credits_attempted_year_2,
            "Number of Credits Earned Year 2": self.number_of_credits_earned_year_2(
                max_value=number_of_credits_attempted_year_2
            ),
            "Number of Credits Attempted Year 3": number_of_credits_attempted_year_3,
            "Number of Credits Earned Year 3": self.number_of_credits_earned_year_3(
                max_value=number_of_credits_attempted_year_3
            ),
            "Number of Credits Attempted Year 4": number_of_credits_attempted_year_4,
            "Number of Credits Earned Year 4": self.number_of_credits_earned_year_4(
                max_value=number_of_credits_attempted_year_4
            ),
            "Gateway Math Status": self.gateway_math_status(),
            "Gateway English Status": self.gateway_english_status(),
            "AttemptedGatewayMathYear1": self.attempted_gateway_math_year_1(),
            "AttemptedGatewayEnglishYear1": self.attempted_gateway_english_year_1(),
            "CompletedGatewayMathYear1": self.completed_gateway_math_year_1(),
            "CompletedGatewayEnglishYear1": self.completed_gateway_english_year_1(),
            "GatewayMathGradeY1": self.gateway_math_grade_y_1(),
            "GatewayEnglishGradeY1": self.gateway_english_grade_y_1(),
            "AttemptedDevMathY1": self.attempted_dev_math_y_1(),
            "AttemptedDevEnglishY1": self.attempted_dev_english_y_1(),
            "CompletedDevMathY1": self.completed_dev_math_y_1(),
            "CompletedDevEnglishY1": self.completed_dev_english_y_1(),
            "Retention": self.retention(),
            "Persistence": self.persistence(),
            "Years to Bachelors at cohort inst.": self.years_to_bachelors_at_cohort_inst(),
            "Years to Associates or Certificate at cohort inst.": self.years_to_associates_or_certificate_at_cohort_inst(),
            "Years to Bachelor at other inst.": self.years_to_bachelor_at_other_inst(),
            "Years to Associates or Certificate at other inst.": self.years_to_associates_or_certificate_at_other_inst(),
            "Years of Last Enrollment at cohort institution": self.years_of_last_enrollment_at_cohort_institution(),
            "Years of Last Enrollment at other institution": self.years_of_last_enrollment_at_other_institution(),
            "Time to Credential": self.time_to_credential(),
            "Reading Placement": self.reading_placement(),
            "Special Program": self.special_program(),
            "NASPA First-Generation": self.naspa_first_generation(),
            "Incarcerated Status": self.incarcerated_status(),
            "Military Status": self.military_status(),
            "Employment Status": self.employment_status(),
            "Disability Status": self.disability_status(),
            "Foreign Language Completion": self.foreign_language_completion(),
            "First Year to Bachelors at cohort inst.": self.first_year_to_bachelors_at_cohort_inst(),
            "First Year to Associates or Certificate at cohort inst.": self.first_year_to_associates_or_certificate_at_cohort_inst(),
            "First Year to Bachelor at other inst.": self.first_year_to_bachelor_at_other_inst(),
            "First Year to Associates or Certificate at other inst.": self.first_year_to_associates_or_certificate_at_other_inst(),
            "Most Recent Bachelors at Other Institution STATE": self.most_recent_bachelors_at_other_institution_state(
                _has_enrollment_other_inst
            ),
            "Most Recent Associates or Certificate at Other Institution STATE": self.most_recent_associates_or_certificate_at_other_institution_state(
                _has_enrollment_other_inst
            ),
            "Most Recent Last Enrollment at Other institution STATE": self.most_recent_last_enrollment_at_other_institution_state(
                _has_enrollment_other_inst
            ),
            "First Bachelors at Other Institution STATE": self.first_bachelors_at_other_institution_state(
                _has_enrollment_other_inst
            ),
            "First Associates or Certificate at Other Institution STATE": self.first_associates_or_certificate_at_other_institution_state(
                _has_enrollment_other_inst
            ),
            "Most Recent Bachelors at Other Institution CARNEGIE": self.most_recent_bachelors_at_other_institution_carnegie(
                _has_enrollment_other_inst
            ),
            "Most Recent Associates or Certificate at Other Institution CARNEGIE": self.most_recent_associates_or_certificate_at_other_institution_carnegie(
                _has_enrollment_other_inst
            ),
            "Most Recent Last Enrollment at Other institution CARNEGIE": self.most_recent_last_enrollment_at_other_institution_carnegie(
                _has_enrollment_other_inst
            ),
            "First Bachelors at Other Institution CARNEGIE": self.first_bachelors_at_other_institution_carnegie(
                _has_enrollment_other_inst
            ),
            "First Associates or Certificate at Other Institution CARNEGIE": self.first_associates_or_certificate_at_other_institution_carnegie(
                _has_enrollment_other_inst
            ),
            "Most Recent Bachelors at Other Institution LOCALE": self.most_recent_bachelors_at_other_institution_locale(
                _has_enrollment_other_inst
            ),
            "Most Recent Associates or Certificate at Other Institution LOCALE": self.most_recent_associates_or_certificate_at_other_institution_locale(
                _has_enrollment_other_inst
            ),
            "Most Recent Last Enrollment at Other institution LOCALE": self.most_recent_last_enrollment_at_other_institution_locale(
                _has_enrollment_other_inst
            ),
            "First Bachelors at Other Institution LOCALE": self.first_bachelors_at_other_institution_locale(
                _has_enrollment_other_inst
            ),
            "First Associates or Certificate at Other Institution LOCALE": self.first_associates_or_certificate_at_other_institution_locale(
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
        return self.numerify("#####!")

    def institution_id(self) -> str:
        return self.numerify("#####!")

    # Returns a string in the format "YYYY-YY" (e.g. "2010-11"), representing a cohort year,
    # where the first year is the start year (a random date between min_yr and max_yr,
    # or min_yr and now if the current date if not provided). The second year is the first year + 1.
    def cohort(self, min_yr: int = 2010, max_yr: t.Optional[int] = None) -> str:
        _end_date = date(max_yr, 1, 1) if max_yr is not None else "today"
        start_dt: date = self.generator.date_between(
            start_date=date(min_yr, 1, 1), end_date=_end_date
        )
        start_yr = start_dt.year
        end_yr = f"{start_yr + 1}"[2:]
        return f"{start_yr}-{end_yr}"

    def cohort_term(self) -> str:
        return self.random_element(["FALL", "WINTER", "SPRING", "SUMMER"])

    def student_age(self) -> str:
        return self.random_element(["20 AND YOUNGER", ">20 - 24", "OLDER THAN 24"])

    def enrollment_type(self) -> str:
        return self.random_element(["FIRST-TIME", "RE-ADMIT", "TRANSFER-IN"])

    def enrollment_intensity_first_term(self) -> str:
        return self.random_element(["FULL-TIME", "PART-TIME"])

    def math_placement(self) -> str:
        return self.random_element(["C", "N"])

    def english_placement(self) -> str:
        return self.random_element(["C", "N"])

    def dual_and_summer_enrollment(self) -> str:
        return self.random_element(["DE", "SE", "DS"])

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

    def first_gen(self) -> str:
        return self.random_element(["P", "C", "A", "B"])

    def pell_status_first_year(self) -> str:
        return self.random_element(["Y", "N"])

    def attendance_status_term_1(
        self,
        enrollment_type: t.Optional[str] = None,
        enrollment_intensity_first_term: t.Optional[str] = None,
    ) -> str:
        if not enrollment_type:
            enrollment_type = self.enrollment_type()
        if not enrollment_intensity_first_term:
            enrollment_intensity_first_term = self.enrollment_intensity_first_term()
        return (
            f"{enrollment_type} {enrollment_intensity_first_term}".title()
            # PDP's string casing is frustratingly inconsistent
            .replace("Admit", "admit")
        )

    def credential_type_sought_year_1(self) -> str:
        return self.random_element(
            [
                "Less than one-year certificate, less than Associate degree",
                "One to two year certificate, less than Associate degree",
                "Two to four year certificate, less than Bachelor's degree",
                "Undergraduate Certificate or Diploma Program",
                "Associate Degree",
                "Bachelor's Degree",
                "Post Baccalaureate Certificate",
                "Master's Degree",
                "Doctoral Degree",
                "First Professional Degree",
                "Graduate/Professional Certificate",
                "Non- Credential Program (Preparatory Coursework/Teach Certification)",
                "Missing",
            ]
        )

    def program_of_study_term_1(self) -> str:
        # TODO: make this six-digit CIP code more realistic
        return self.numerify("##.####")

    def program_of_study_year_1(self) -> str:
        # TODO: make this six-digit CIP code more realistic
        return self.numerify("##.####")

    def _gpa(self) -> float:
        return self.generator.pyfloat(  # type: ignore
            min_value=0.0, max_value=4.0, right_digits=2
        )

    def gpa_group_term_1(self) -> float:
        return self._gpa()

    def gpa_group_year_1(self) -> float:
        return self._gpa()

    def _number_of_credits(
        self, min_value: float = 1.0, max_value: float = 20.0
    ) -> float:
        return self.generator.pyfloat(  # type: ignore
            min_value=min_value,
            max_value=max(max_value, min_value + 1e-3),
            right_digits=1,
        )

    def number_of_credits_attempted_year_1(self) -> float:
        return self._number_of_credits(min_value=1.0)

    def number_of_credits_earned_year_1(self, max_value: float = 20.0) -> float:
        return self._number_of_credits(max_value=max_value)

    def number_of_credits_attempted_year_2(self) -> float:
        return self._number_of_credits()

    def number_of_credits_earned_year_2(self, max_value: float = 20.0) -> float:
        return self._number_of_credits(max_value=max_value)

    def number_of_credits_attempted_year_3(self) -> float:
        return self._number_of_credits()

    def number_of_credits_earned_year_3(self, max_value: float = 20.0) -> float:
        return self._number_of_credits(max_value=max_value)

    def number_of_credits_attempted_year_4(self) -> float:
        return self._number_of_credits()

    def number_of_credits_earned_year_4(self, max_value: float = 20.0) -> float:
        return self._number_of_credits(max_value=max_value)

    def gateway_math_status(self) -> str:
        return self.random_element(["R", "N"])

    def gateway_english_status(self) -> str:
        return self.random_element(["R", "N"])

    def attempted_gateway_math_year_1(self) -> str:
        return self.random_element(["Y", "N"])

    def attempted_gateway_english_year_1(self) -> str:
        return self.random_element(["Y", "N"])

    def completed_gateway_math_year_1(self) -> str:
        return self.random_element(["C", "D", "NA"])

    def completed_gateway_english_year_1(self) -> str:
        return self.random_element(["C", "D", "NA"])

    def _grade(self) -> str:
        # TODO: use weighting for more realistic distribution?
        # find out if grades can ever be given as continuous values
        return self.random_element(
            ["0", "1", "2", "3", "4", "P", "F", "I", "W", "A", "M", "O"]
        )

    def gateway_math_grade_y_1(self) -> str:
        return self._grade()

    def gateway_english_grade_y_1(self) -> str:
        return self._grade()

    def attempted_dev_math_y_1(self) -> str:
        return self.random_element(["Y", "N", "NA"])

    def attempted_dev_english_y_1(self) -> str:
        return self.random_element(["Y", "N", "NA"])

    def completed_dev_math_y_1(self) -> str:
        return self.random_element(["Y", "N", "NA"])

    def completed_dev_english_y_1(self) -> str:
        return self.random_element(["Y", "N", "NA"])

    def retention(self) -> bool:
        return self.generator.pybool()  # type: ignore

    def persistence(self) -> bool:
        return self.generator.pybool()  # type: ignore

    def _years_to_of(self, min_value: int = 0, max_value: int = 7) -> int:
        return self.random_int(min=min_value, max=max_value)

    def years_to_bachelors_at_cohort_inst(self) -> int:
        return self._years_to_of()

    def years_to_associates_or_certificate_at_cohort_inst(self) -> int:
        return self._years_to_of()

    def years_to_bachelor_at_other_inst(
        self, has_enrollment: bool = True
    ) -> t.Optional[int]:
        return self._years_to_of() if has_enrollment else None

    def years_to_associates_or_certificate_at_other_inst(
        self, has_enrollment: bool = True
    ) -> t.Optional[int]:
        return self._years_to_of() if has_enrollment else None

    def years_of_last_enrollment_at_cohort_institution(self) -> int:
        return self._years_to_of()

    def years_of_last_enrollment_at_other_institution(
        self, has_enrollment: bool = True
    ) -> t.Optional[int]:
        return self._years_to_of() if has_enrollment else None

    def time_to_credential(self) -> float:
        return self.generator.pyfloat(  # type: ignore
            min_value=1.0, max_value=20.0, right_digits=0
        )

    def reading_placement(self) -> str:
        return self.random_element(["C", "N"])

    def special_program(self) -> t.Optional[str]:
        # TODO: come up with something here?
        return None

    def naspa_first_generation(self) -> t.Optional[str]:
        return self.random_element(["-1", "0", "1", "2", "3", "4", "5", "6"])

    def incarcerated_status(self) -> t.Optional[str]:
        return self.random_element(["Y", "P", "N"])

    def military_status(self) -> t.Optional[str]:
        return self.random_element(["-1", "0", "1", "2"])

    def employment_status(self) -> t.Optional[str]:
        return self.random_element(["-1", "0", "1", "2", "3", "4"])

    def disability_status(self) -> t.Optional[str]:
        return self.random_element(["Y", "N"])

    def foreign_language_completion(self) -> t.Optional[str]:
        # TODO: come up with something here?
        return None

    def first_year_to_bachelors_at_cohort_inst(self) -> int:
        return self._years_to_of()

    def first_year_to_associates_or_certificate_at_cohort_inst(self) -> int:
        return self._years_to_of()

    def first_year_to_bachelor_at_other_inst(
        self, has_enrollment: bool = True
    ) -> t.Optional[int]:
        return self._years_to_of() if has_enrollment else None

    def first_year_to_associates_or_certificate_at_other_inst(
        self, has_enrollment: bool = True
    ) -> t.Optional[int]:
        return self._years_to_of() if has_enrollment else None

    def _institution_state(self) -> str:
        return self.generator.state_abbr(include_freely_associated_states=False)  # type: ignore

    def most_recent_bachelors_at_other_institution_state(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_state() if has_enrollment else None

    def most_recent_associates_or_certificate_at_other_institution_state(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_state() if has_enrollment else None

    def most_recent_last_enrollment_at_other_institution_state(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_state() if has_enrollment else None

    def first_bachelors_at_other_institution_state(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_state() if has_enrollment else None

    def first_associates_or_certificate_at_other_institution_state(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_state() if has_enrollment else None

    def _institution_carnegie(self) -> str:
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

    def most_recent_bachelors_at_other_institution_carnegie(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_carnegie() if has_enrollment else None

    def most_recent_associates_or_certificate_at_other_institution_carnegie(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_carnegie() if has_enrollment else None

    def most_recent_last_enrollment_at_other_institution_carnegie(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_carnegie() if has_enrollment else None

    def first_bachelors_at_other_institution_carnegie(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_carnegie() if has_enrollment else None

    def first_associates_or_certificate_at_other_institution_carnegie(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_carnegie() if has_enrollment else None

    def _institution_locale(self) -> str:
        return self.random_element(["URBAN", "SUBURB", "TOWN/RURAL"])

    def most_recent_bachelors_at_other_institution_locale(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_locale() if has_enrollment else None

    def most_recent_associates_or_certificate_at_other_institution_locale(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_locale() if has_enrollment else None

    def most_recent_last_enrollment_at_other_institution_locale(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_locale() if has_enrollment else None

    def first_bachelors_at_other_institution_locale(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_locale() if has_enrollment else None

    def first_associates_or_certificate_at_other_institution_locale(
        self, has_enrollment: bool = True
    ) -> t.Optional[str]:
        return self._institution_locale() if has_enrollment else None
