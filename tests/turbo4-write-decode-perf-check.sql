.bail on

CREATE TEMP TABLE expected_write_cases (op_params TEXT PRIMARY KEY);
INSERT INTO expected_write_cases VALUES
('t4_write_decode_type_idx=i32,cache_rows=8192,row=4096,pattern=2'),
('t4_write_decode_type_idx=i64,cache_rows=8192,row=4096,pattern=2');

CREATE TEMP TABLE write_perf_gate (passed INTEGER NOT NULL CHECK (passed = 1));
INSERT INTO write_perf_gate
SELECT CASE WHEN
    (SELECT count(*) FROM test_backend_ops) = 2
    AND (SELECT count(DISTINCT op_params) FROM test_backend_ops) = 2
    AND NOT EXISTS (
        SELECT 1 FROM test_backend_ops t
        LEFT JOIN expected_write_cases e ON t.op_params = e.op_params
        WHERE COALESCE(
            e.op_params IS NOT NULL
            AND t.backend_name = 'MTL0' AND t.op_name = 'SET_ROWS_TURBO4'
            AND t.test_mode = 'perf' AND t.supported = 1 AND t.passed = 1
            AND t.error_message = ''
            AND typeof(t.time_us) IN ('real', 'integer')
            AND t.time_us > 0 AND t.time_us - t.time_us = 0
            AND typeof(t.n_runs) = 'integer' AND t.n_runs > 0,
            0) = 0
    )
THEN 1 ELSE 0 END;

SELECT '2/2 writer performance cases validated' FROM write_perf_gate;
