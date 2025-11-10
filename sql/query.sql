-- select count(distinct(m.object_id)), count(m.object_id)
select m.object_id
from maps_apr25_test as m
join uvs_apr25_test as u
on m.freq = u.freq
and m.object_name = u.object_name
and m.obs_date = u.obs_date
where u.uv_quality = 0; 
-- 185

-- SET map_quality = 0
-- FROM uvs_apr25_test AS u
-- WHERE u.uv_quality = 0
--   AND m.freq = u.freq
--   AND m.object_name = u.object_name
--   AND m.obs_date = u.obs_date;

-- check inverse
