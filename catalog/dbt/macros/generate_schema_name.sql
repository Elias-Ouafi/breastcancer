{#
  dbt's default prefixes a custom schema with the target schema (`main_stg`). The
  catalogue's schemas are part of its interface -- `SELECT * FROM mart.dim_patient`
  is what the docs, the example queries and the CLI use -- so they are taken as-is.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ custom_schema_name if custom_schema_name is not none else target.schema }}
{%- endmacro %}
