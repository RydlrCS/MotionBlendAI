-- macros/velocity.sql
-- Calculate L2 velocity between frames

{% macro l2_velocity(x, x_prev, y, y_prev, z, z_prev) %}
    SQRT(
        POW({{ x }} - {{ x_prev }}, 2) + 
        POW({{ y }} - {{ y_prev }}, 2) + 
        POW({{ z }} - {{ z_prev }}, 2)
    )
{% endmacro %}

{% macro l2_acceleration(v_curr, v_prev) %}
    ABS({{ v_curr }} - {{ v_prev }})
{% endmacro %}
