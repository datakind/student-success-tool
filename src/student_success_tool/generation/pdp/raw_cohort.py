import typing as t
from datetime import date

from faker.providers import BaseProvider


class Provider(BaseProvider):
    def student_guid(self) -> str:
        return self.numerify("#####!")  # type: ignore

    def institution_id(self) -> str:
        return self.numerify("#####!")  # type: ignore

    def cohort(self, min_yr: int = 2000, max_yr: t.Optional[int] = None) -> str:
        start_dt = self.generator.date_between(start_date=date(min_yr, 1, 1))
        end_dt = start_dt.replace(year=start_dt.year + 1)
        return f"{start_dt:%Y}-{end_dt:%Y}"

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

    def attendance_status_term_1(self) -> str:
        return self.random_element(
            [
                "First-Time Full-Time",
                "First-Time Part-Time",
                "First-Time Unknown",
                "Transfer-In Full-Time",
                "Transfer-In Part-Time",
                "Transfer-In Unknown",
                "Re-admit Full-Time",
                "Re-admit Part-Time",
                "Re-admit Unknown",
            ]
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

    def _gpa(self) -> float:
        return self.generator.pyfloat(  # type: ignore
            min_value=0.0, max_value=4.0, right_digits=2
        )

    def gpa_group_term_1(self) -> float:
        return self._gpa()

    def gpa_group_year_1(self) -> float:
        return self._gpa()

    def _number_of_credits_attempted(self, min_value: float = 1.0) -> float:
        return self.generator.pyfloat(  # type: ignore
            min_value=min_value, max_value=20.0, right_digits=1
        )

    def number_of_credits_attempted_year_1(self) -> float:
        return self._number_of_credits_attempted(min_value=1.0)

    def number_of_credits_earned_year_1(self) -> float:
        return self._number_of_credits_attempted()

    def number_of_credits_attempted_year_2(self) -> float:
        return self._number_of_credits_attempted()

    def number_of_credits_earned_year_2(self) -> float:
        return self._number_of_credits_attempted()

    def number_of_credits_attempted_year_3(self) -> float:
        return self._number_of_credits_attempted()

    def number_of_credits_earned_year_3(self) -> float:
        return self._number_of_credits_attempted()

    def number_of_credits_attempted_year_4(self) -> float:
        return self._number_of_credits_attempted()

    def number_of_credits_earned_year_4(self) -> float:
        return self._number_of_credits_attempted()

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
        return ""

    def completed_dev_english_y_1(self) -> str:
        return self.random_element(["Y", "N", "NA"])

    def retention(self) -> bool:
        return self.generator.pybool()  # type: ignore

    def persistence(self) -> bool:
        return self.generator.pybool()  # type: ignore

    def _years_to_of(self) -> int:
        return self.random_int(min=0, max=7)

    def years_to_bachelors_at_cohort_inst(self) -> int:
        return self._years_to_of()

    def years_to_associates_or_certificate_at_cohort_inst(self) -> int:
        return self._years_to_of()

    def years_to_bachelor_at_other_inst(self) -> int:
        return self._years_to_of()

    def years_to_associates_or_certificate_at_other_inst(self) -> int:
        return self._years_to_of()

    def years_of_last_enrollment_at_cohort_institution(self) -> int:
        return self._years_to_of()

    def years_of_last_enrollment_at_other_institution(self) -> int:
        return self._years_to_of()

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

    def first_year_to_bachelor_at_other_inst(self) -> int:
        return self._years_to_of()

    def first_year_to_associates_or_certificate_at_other_inst(self) -> int:
        return self._years_to_of()

    def program_of_study_year_1(self) -> str:
        # TODO: make this six-digit CIP code more realistic
        return self.numerify("##.####")

    def _institution_state(self) -> str:
        return self.generator.state_abbr(include_freely_associated_states=False)  # type: ignore

    def most_recent_bachelors_at_other_institution_state(self) -> str:
        return self._institution_state()

    def most_recent_associates_or_certificate_at_other_institution_state(self) -> str:
        return self._institution_state()

    def most_recent_last_enrollment_at_other_institution_state(self) -> str:
        return self._institution_state()

    def first_bachelors_at_other_institution_state(self) -> str:
        return self._institution_state()

    def first_associates_or_certificate_at_other_institution_state(self) -> str:
        return self._institution_state()

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

    def most_recent_bachelors_at_other_institution_carnegie(self) -> str:
        return self._institution_carnegie()

    def most_recent_associates_or_certificate_at_other_institution_carnegie(
        self,
    ) -> str:
        return self._institution_carnegie()

    def most_recent_last_enrollment_at_other_institution_carnegie(self) -> str:
        return self._institution_carnegie()

    def first_bachelors_at_other_institution_carnegie(self) -> str:
        return self._institution_carnegie()

    def first_associates_or_certificate_at_other_institution_carnegie(self) -> str:
        return self._institution_carnegie()

    def _institution_locale(self) -> str:
        return self.random_element(["URBAN", "SUBURB", "TOWN/RURAL"])

    def most_recent_bachelors_at_other_institution_locale(self) -> str:
        return self._institution_locale()

    def most_recent_associates_or_certificate_at_other_institution_locale(self) -> str:
        return self._institution_locale()

    def most_recent_last_enrollment_at_other_institution_locale(self) -> str:
        return self._institution_locale()

    def first_bachelors_at_other_institution_locale(self) -> str:
        return self._institution_locale()

    def first_associates_or_certificate_at_other_institution_locale(self) -> str:
        return self._institution_locale()
