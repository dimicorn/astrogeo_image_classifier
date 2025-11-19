\copy (
    SELECT DISTINCT
        t.object_name,
        CASE
            WHEN t.freq BETWEEN 1.8e9 AND 2.8e9 THEN 'S'
            WHEN t.freq BETWEEN 2.8e9 AND 7e9 THEN 'C'
            WHEN t.freq BETWEEN 7e9 AND 9e9 THEN 'X'
            ELSE 'N/A'
        END AS freq_band
    FROM maps_apr25_original AS t
    WHERE
        (
            t.freq BETWEEN 1.8e9 AND 2.8e9
            AND NOT EXISTS (
                SELECT 1
                FROM maps_apr25_original AS x
                WHERE x.object_name = t.object_name
                    AND x.freq BETWEEN 1.8e9 AND 2.8e9
                    AND x.map_quality = 1
            )
        )
        OR
        (
            t.freq BETWEEN 2.8e9 AND 7e9
            AND NOT EXISTS (
                SELECT 1
                FROM maps_apr25_original AS x
                WHERE x.object_name = t.object_name
                    AND x.freq BETWEEN 2.8e9 AND 7e9
                    AND x.map_quality = 1
            )
        )
        OR
        (
            t.freq BETWEEN 7e9 AND 9e9
            AND NOT EXISTS (
                SELECT 1
                FROM maps_apr25_original AS x
                WHERE x.object_name = t.object_name
                    AND x.freq BETWEEN 7e9 AND 9e9
                    AND x.map_quality = 1
            )
        )
) to 'csv/no_good_obs.csv' with csv header;