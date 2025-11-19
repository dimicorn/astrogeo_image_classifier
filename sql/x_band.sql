-- Interactive mode only
\copy (
  select object_name, obs_date, freq, file_name, comment
  from maps_apr25_new
  where map_quality = 0 and freq >= 7e9 and freq <= 9e9
) to 'csv/maps_apr25_new_x_band.csv' with csv header;

\copy (
  select object_name, obs_date, freq, file_name, comment
  from maps_apr25_original
  where map_quality = 0 and freq >= 7e9 and freq <= 9e9
) to 'csv/maps_apr25_orig_x_band.csv' with csv header;