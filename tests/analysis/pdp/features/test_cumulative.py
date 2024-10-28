# TODO: test this functionality extensively

# @pytest.mark.parametrize(
#     ["df", "grp_cols", "term_rank_col", "exp"],
#     [
#         (
#             pd.DataFrame(
#                 {
#                     "iid": ["x", "x", "x", "x", "x", "x", "x"],
#                     "sid": ["a", "a", "b", "b", "a", "a", "b"],
#                     "term_rank": [1, 3, 0, 1, 4, 6, 5],
#                 }
#             ),
#             ["iid", "sid"],
#             "term_rank",
#             pd.Series([0.0, 0.333333, 0.0, 0.0, 0.25, 0.333333, 0.5]),
#         ),
#     ],
# )
# def test_compute_cumfrac_terms_unenrolled(df, grp_cols, term_rank_col, exp):
#     obs = features.compute_cumfrac_terms_unenrolled(
#         df, grp_cols=grp_cols, term_rank_col=term_rank_col
#     )
#     assert isinstance(obs, pd.Series) and not obs.empty
#     assert pd.testing.assert_series_equal(obs, exp) is None  # raises error if not equal
