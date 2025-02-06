import argparse
import logging
import pathlib
import sys

import faker
import pandas as pd

from student_success_tool.generation import pdp


def main():
    args = add_and_parse_arguments()

    logging.basicConfig(level=logging.INFO)

    faker.Faker.seed(args.seed)
    FAKER = faker.Faker()
    FAKER.add_provider(pdp.raw_cohort.Provider)
    FAKER.add_provider(pdp.raw_course.Provider)

    # institution_id must be the same for all records.
    institution_id = FAKER.numerify("#####!")
    student_guids = [FAKER.unique.student_guid() for _ in range(args.num_students)]
    cohort_records = [
        FAKER.raw_cohort_record(
            normalize_col_names=args.normalize_col_names,
            institution_id=institution_id,
            student_guid=student_guids[i],
        )
        for i in range(args.num_students)
    ]
    course_records = [
        FAKER.raw_course_record(
            cohort_record, normalize_col_names=args.normalize_col_names
        )
        for cohort_record in cohort_records
        for _ in range(
            FAKER.randomize_nb_elements(args.avg_num_courses_per_student, min=1)
        )
    ]
    df_cohort = pd.DataFrame(cohort_records)
    df_course = pd.DataFrame(course_records)

    logging.info(
        "generated %s cohort records and %s course records",
        len(cohort_records),
        len(course_records),
    )
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        df_cohort.to_csv(
            args.save_dir / "INSTXYZ_STUDENT_SEMESTER_AR_DEIDENTIFIED.csv",
            header=True,
            index=False,
        )
        df_course.to_csv(
            args.save_dir / "INSTXYZ_COURSE_LEVEL_AR_DEID.csv",
            header=True,
            index=False,
        )
        logging.info("datasets saved to disk at %s", args.save_dir)


def add_and_parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num-students", type=int, default=3)
    parser.add_argument("--avg-num-courses-per-student", type=int, default=5)
    parser.add_argument("--normalize-col-names", action="store_true", default=False)
    parser.add_argument("--save-dir", type=pathlib.Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    sys.exit(main())
