# Technical Project Brief: Natural Language Interface to ATNF Pulsar Catalogue

**Project Title:** ATNF-Chat: LLM-Powered Conversational Interface for Pulsar Catalogue Queries

**Version:** 1.1  
**Date:** December 17, 2025  
**Author:** Tom Kimpson  
**Reviewer:** External Technical Review (incorporated)

---

## Executive Summary

This document outlines the technical design for a natural language interface to the ATNF Pulsar Catalogue, enabling researchers to query pulsar data, generate visualizations, and perform analyses through conversational interactions rather than SQL queries or Python scripts. The system will leverage Large Language Models (LLMs) with function calling capabilities to translate natural language into structured catalogue queries and data operations.

**Key Updates in v1.1:**
- Dynamic catalogue version handling (no hardcoded versions)
- Strict query DSL with validation layer
- Preference for ATNF-native derived parameters
- Enhanced result provenance and null handling
- Automated schema grounding from ATNF documentation
- Expanded benchmark suite including failure cases

---

## 1. Project Motivation

### 1.1 Current State
The ATNF Pulsar Catalogue is the authoritative source for pulsar parameters, containing:
- Regularly updated collection of known pulsars with comprehensive metadata
- 100+ parameters per pulsar (timing, astrometric, binary, derived)
- Regular updates via CSIRO maintenance
- Access via: web interface, command-line tool (psrcat), Python wrapper (psrqpy)
- Versioned releases available through CSIRO Data Access Portal

**Note:** Catalogue statistics (pulsar counts, version numbers) are determined at runtime from the locally cached catalogue to ensure accuracy. The system reports the detected catalogue version and snapshot date in responses.

### 1.2 Problem Statement
Current access methods require:
- Knowledge of parameter naming conventions (e.g., F0, F1, P0, DM)
- Understanding of query syntax (SQL-like conditions)
- Manual data manipulation for visualizations
- Context switching between documentation and implementation
- Awareness of data quality flags and missingness patterns

### 1.3 Proposed Solution
A conversational AI agent that:
- Accepts natural language queries ("Show me all millisecond pulsars in globular clusters")
- Translates to validated catalogue operations with provenance tracking
- Generates publication-quality visualizations
- Provides contextual explanations with uncertainty handling
- Maintains conversation history for iterative refinement
- Ensures scientific safety through explicit null handling and result validation

---

## 2. System Architecture

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                     User Interface                       │
│  (Web Chat / CLI / Jupyter Widget / API Endpoint)       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  LLM Orchestration Layer                 │
│  - Intent Classification                                 │
│  - Function/Tool Selection                               │
│  - Query DSL Validation                                  │
│  - Response Generation with Provenance                   │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┬────────────────┐
         ▼                       ▼                ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  Query Tools     │   │  Analysis Tools  │   │  Viz Tools       │
│  - DSL Parser    │   │  - Statistics    │   │  - matplotlib    │
│  - Query Validator│  │  - Correlations  │   │  - plotly        │
│  - psrqpy wrapper│   │  - Clustering    │   │  - custom plots  │
└──────────────────┘   └──────────────────┘   └──────────────────┘
         │                       │                │
         └───────────┬───────────┴────────────────┘
                     ▼
         ┌────────────────────────┐
         │  ATNF Catalogue Data   │
         │  - Versioned snapshots │
         │  - psrqpy interface    │
         │  - Schema metadata     │
         │  - Periodic updates    │
         └────────────────────────┘
```

### 2.2 Component Breakdown

#### 2.2.1 User Interface Options

**Option A: Web Application (Recommended)**
- **Technology:** React/Next.js frontend, FastAPI backend
- **Advantages:** 
  - Familiar chat interface (similar to ChatGPT/Claude)
  - Easy to share and deploy
  - Supports rich visualizations in-browser
  - Authentication/user sessions
- **Implementation:**
  - Real-time streaming responses
  - Interactive plot rendering (Plotly/Altair)
  - Code export functionality
  - Chat history persistence
  - Query plan visualization

**Option B: Jupyter Widget**
- **Technology:** ipywidgets, Panel, or Gradio
- **Advantages:**
  - Integrates into existing research workflow
  - Direct access to Python environment
  - Easy to combine with custom analysis
- **Implementation:**
  - Widget displays in notebook cell
  - Outputs rendered directly below
  - Can export generated code to cells

**Option C: Command-Line Interface**
- **Technology:** Rich/Textual for TUI
- **Advantages:**
  - Lightweight, no browser required
  - Scriptable for automation
  - SSH-friendly
- **Implementation:**
  - Interactive REPL-style interface
  - ASCII table rendering
  - Plot generation saves to files

**Recommendation:** Implement Web Application first with API backend that enables other interfaces later.

#### 2.2.2 LLM Selection and Configuration

**Primary LLM Candidates:**

1. **Anthropic Claude Sonnet 4.5** (Recommended)
   - **Strengths:**
     - Excellent function calling accuracy
     - Strong scientific/technical understanding
     - Large context window (200k tokens)
     - Good at interpreting domain-specific terminology
   - **Configuration:**
     - Temperature: 0.1 (precise, deterministic)
     - Max tokens: 4096
     - System prompt with catalogue schema
   - **Source:** Anthropic model documentation
   
2. **OpenAI GPT-4o**
   - **Strengths:**
     - Mature function calling
     - Good code generation
     - Faster inference
   - **Limitations:**
     - Smaller context window (128k tokens)
     - More expensive for extended use
   - **Source:** OpenAI platform documentation

3. **Open-Source Alternative: Llama 3.3 70B**
   - **Strengths:**
     - Self-hostable
     - No API costs
     - Privacy control
   - **Limitations:**
     - Requires significant compute (A100/H100)
     - Function calling less reliable
     - May need fine-tuning on astronomy corpus

**Architecture Decision: Hybrid Approach**
- Primary: Claude API for production
- Fallback: Local Llama model via Ollama/vLLM
- Allows cost control and offline operation

**Critical Design Note:** The model matters less than the tool contract, validation, and schema grounding. Invest engineering time in guardrails rather than model selection.

### 2.3 Data Layer Design

#### 2.3.1 Catalogue Access Strategy

**Approach 1: Direct psrqpy Integration (Recommended for MVP)**
```python
from psrqpy import QueryATNF
import datetime

class CatalogueInterface:
    def __init__(self):
        # Load entire catalogue on init
        self.query = QueryATNF(loadfromdb=True)
        self.df = self.query.pandas  # Full DataFrame
        
        # Extract and store catalogue metadata
        self.catalogue_metadata = {
            'version': self._detect_catalogue_version(),
            'snapshot_date': datetime.datetime.now().isoformat(),
            'total_pulsars': len(self.df),
            'measured_parameters': self._count_measured_params()
        }
        
    def _detect_catalogue_version(self):
        """Extract catalogue version from psrqpy query metadata"""
        # psrqpy stores catalogue version in query object
        return self.query.catalogue_version if hasattr(self.query, 'catalogue_version') else 'unknown'
    
    def _count_measured_params(self):
        """Count non-null values for each parameter"""
        return {col: self.df[col].notna().sum() for col in self.df.columns}
        
    def get_catalogue_info(self):
        """Return catalogue metadata for system prompt and responses"""
        return self.catalogue_metadata
    
    def query_pulsars_dsl(self, query_dsl: dict):
        """Execute query using validated DSL (see section 2.3.3)"""
        # Convert DSL to psrqpy condition string
        condition = self._dsl_to_condition(query_dsl)
        params = query_dsl.get('select_fields', None)
        
        q = QueryATNF(params=params, condition=condition)
        result_df = q.pandas
        
        # Add result provenance
        provenance = {
            'query_dsl': query_dsl,
            'condition_string': condition,
            'result_count': len(result_df),
            'null_counts': {col: result_df[col].isna().sum() for col in result_df.columns},
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return result_df, provenance
```

**Advantages:**
- Leverages mature, tested library
- Handles all parameter conversions
- Built-in uncertainty handling
- Automatic catalogue updates (with version tracking)

**Approach 2: Local SQLite Database**
```sql
-- Schema design with versioning
CREATE TABLE catalogue_versions (
    id INTEGER PRIMARY KEY,
    version TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    source_url TEXT,
    total_pulsars INTEGER
);

CREATE TABLE pulsars (
    jname TEXT PRIMARY KEY,
    psrb TEXT,
    rajd REAL,  -- Right Ascension (degrees)
    decjd REAL, -- Declination (degrees)
    p0 REAL,    -- Period (s)
    f0 REAL,    -- Frequency (Hz)
    f1 REAL,    -- Frequency derivative
    dm REAL,    -- Dispersion measure
    -- ATNF-native derived fields
    bsurf REAL, -- Surface magnetic field (Gauss)
    edot REAL,  -- Spin-down energy loss rate (erg/s)
    -- ... 100+ additional columns
    catalogue_version_id INTEGER,
    FOREIGN KEY (catalogue_version_id) REFERENCES catalogue_versions(id)
);

CREATE TABLE parameter_quality (
    pulsar_jname TEXT,
    param_name TEXT,
    value REAL,
    error REAL,
    reference TEXT,
    is_limit INTEGER,  -- 1 if upper/lower limit
    quality_flag TEXT, -- From ATNF quality indicators
    FOREIGN KEY (pulsar_jname) REFERENCES pulsars(jname)
);
```

**Advantages:**
- Fast querying with indexes
- Complex JOIN operations
- Easier to add custom annotations
- Version control of data snapshots
- Explicit tracking of data quality

**Recommendation:** Start with psrqpy for MVP, migrate to SQLite if custom schema or complex versioning needed.

#### 2.3.2 Parameter Mapping Schema

**Automated Schema Grounding Strategy:**

Instead of hand-curating mappings, generate them from ATNF documentation:

```python
class SchemaGroundingPack:
    """
    Automatically generated from ATNF parameter documentation
    Source: https://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_help.html
    """
    
    def __init__(self):
        self.canonical_params = self._load_atnf_parameters()
        self.human_aliases = self._load_curated_aliases()
        self.unit_registry = self._initialize_unit_registry()
        
    def _load_atnf_parameters(self):
        """
        Load official ATNF parameter definitions
        Could be scraped from documentation or loaded from JSON
        """
        return {
            'P0': {
                'description': 'Barycentric period of the pulsar (s)',
                'unit': 'seconds',
                'type': 'measured',
                'typical_range': [0.001, 10.0],
                'related_params': ['F0', 'P1']
            },
            'P1': {
                'description': 'Time derivative of barycentric period',
                'unit': 'dimensionless',
                'type': 'measured',
                'typical_range': [1e-21, 1e-12],
                'related_params': ['P0', 'F1']
            },
            'F0': {
                'description': 'Barycentric rotation frequency (Hz)',
                'unit': 'hertz',
                'type': 'measured',
                'typical_range': [0.1, 1000.0],
                'related_params': ['P0', 'F1']
            },
            'F1': {
                'description': 'Time derivative of barycentric rotation frequency',
                'unit': 's^-2',
                'type': 'measured',
                'typical_range': [-1e-9, -1e-15],
                'related_params': ['F0', 'P1']
            },
            'DM': {
                'description': 'Dispersion measure (pc cm^-3)',
                'unit': 'pc cm^-3',
                'type': 'measured',
                'typical_range': [0.0, 2000.0]
            },
            'BSURF': {
                'description': 'Surface magnetic field strength (Gauss)',
                'unit': 'Gauss',
                'type': 'derived_atnf',  # ATNF provides this
                'formula': 'sqrt(3 * c^3 * I * P * P1 / (8 * pi^2 * R^6))',
                'typical_range': [1e8, 1e14]
            },
            'EDOT': {
                'description': 'Spin-down energy loss rate (erg/s)',
                'unit': 'erg/s',
                'type': 'derived_atnf',  # ATNF provides this
                'formula': '4 * pi^2 * I * F0^3 * abs(F1)',
                'typical_range': [1e26, 1e38]
            },
            # ... more parameters
        }
    
    def _load_curated_aliases(self):
        """
        Human-maintained alias mappings for natural language
        """
        return {
            'P0': ['period', 'spin period', 'rotation period', 'pulse period'],
            'P1': ['period derivative', 'spin-down rate', 'Pdot'],  # Note: Pdot is alias
            'F0': ['frequency', 'spin frequency', 'rotation frequency'],
            'F1': ['frequency derivative', 'Fdot'],
            'DM': ['dispersion measure', 'dispersion'],
            'BSURF': ['magnetic field', 'surface field', 'B field'],
            'EDOT': ['spin-down luminosity', 'energy loss rate', 'Edot'],
            'BINARY': ['binary', 'binary system', 'companion'],
            'ASSOC': ['association', 'cluster', 'associated with']
        }
    
    def _initialize_unit_registry(self):
        """Use astropy units for all conversions"""
        from astropy import units as u
        
        return {
            'P0': u.second,
            'P1': u.dimensionless_unscaled,
            'F0': u.Hz,
            'F1': u.Hz / u.second,
            'DM': u.pc / u.cm**3,
            'BSURF': u.Gauss,
            'EDOT': u.erg / u.second
        }
    
    def regenerate_mappings(self):
        """
        Call this when ATNF adds new parameters
        Automatically updates LLM context
        """
        mapping = {}
        for param, info in self.canonical_params.items():
            mapping[param] = {
                'canonical': param,
                'aliases': self.human_aliases.get(param, []),
                'unit': info['unit'],
                'description': info['description'],
                'type': info['type']
            }
        
        # Save to JSON for LLM system prompt
        import json
        with open('parameter_mappings.json', 'w') as f:
            json.dump(mapping, f, indent=2)
        
        return mapping
```

**Key Improvements:**
- Canonical parameter codes from official ATNF documentation
- Automatic regeneration when ATNF adds fields
- Distinction between measured and ATNF-derived parameters
- Human aliases maintained separately for flexibility

#### 2.3.3 Query DSL with Validation

**Critical Design Decision:** Use an explicit, validated DSL instead of free-form filter strings.

**DSL Schema:**
```json
{
  "query": {
    "select_fields": ["JNAME", "P0", "DM"],
    "filters": {
      "op": "and",
      "clauses": [
        {
          "field": "P0",
          "cmp": "lt",
          "value": 0.03,
          "unit": "s"
        },
        {
          "field": "ASSOC",
          "cmp": "contains",
          "value": "GC"
        }
      ]
    },
    "order_by": "P0",
    "limit": 100
  }
}
```

**Supported Comparison Operators:**
- `eq`: equals
- `ne`: not equals
- `lt`: less than
- `le`: less than or equal
- `gt`: greater than
- `ge`: greater than or equal
- `contains`: string contains
- `in_range`: value in [min, max]
- `is_null`: field is null/missing
- `not_null`: field is measured

**DSL Validator Implementation:**

```python
from pydantic import BaseModel, Field, validator
from typing import Literal, Union, List, Optional
from enum import Enum

class ComparisonOp(str, Enum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    CONTAINS = "contains"
    IN_RANGE = "in_range"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"

class FilterClause(BaseModel):
    field: str
    cmp: ComparisonOp
    value: Union[float, str, List[float], None] = None
    unit: Optional[str] = None
    
    @validator('field')
    def validate_field(cls, v):
        # Check against known ATNF parameters
        schema = SchemaGroundingPack()
        valid_fields = list(schema.canonical_params.keys())
        if v.upper() not in valid_fields:
            raise ValueError(f"Unknown field: {v}. Valid fields: {valid_fields}")
        return v.upper()
    
    @validator('unit')
    def validate_unit(cls, v, values):
        if v is None:
            return v
        schema = SchemaGroundingPack()
        field = values.get('field')
        expected_unit = schema.canonical_params[field]['unit']
        # Could add unit conversion logic here
        return v

class LogicalOp(str, Enum):
    AND = "and"
    OR = "or"

class FilterGroup(BaseModel):
    op: LogicalOp
    clauses: List[Union[FilterClause, 'FilterGroup']]

FilterGroup.update_forward_refs()

class QueryDSL(BaseModel):
    select_fields: Optional[List[str]] = None
    filters: Optional[FilterGroup] = None
    order_by: Optional[str] = None
    limit: Optional[int] = Field(None, ge=1, le=10000)
    
    @validator('select_fields', each_item=True)
    def validate_select_fields(cls, v):
        schema = SchemaGroundingPack()
        valid_fields = list(schema.canonical_params.keys())
        if v.upper() not in valid_fields:
            raise ValueError(f"Unknown field: {v}")
        return v.upper()
    
    def to_psrqpy_condition(self) -> str:
        """Convert DSL to psrqpy condition string"""
        if self.filters is None:
            return None
        
        def build_condition(filter_group):
            if isinstance(filter_group, FilterClause):
                return self._clause_to_string(filter_group)
            else:
                op_str = " && " if filter_group.op == LogicalOp.AND else " || "
                subconditions = [build_condition(c) for c in filter_group.clauses]
                return f"({op_str.join(subconditions)})"
        
        return build_condition(self.filters)
    
    def _clause_to_string(self, clause: FilterClause) -> str:
        """Convert single clause to psrqpy condition"""
        field = clause.field
        op_map = {
            ComparisonOp.EQ: "==",
            ComparisonOp.NE: "!=",
            ComparisonOp.LT: "<",
            ComparisonOp.LE: "<=",
            ComparisonOp.GT: ">",
            ComparisonOp.GE: ">=",
            ComparisonOp.CONTAINS: "contains",
            ComparisonOp.IS_NULL: "is null",
            ComparisonOp.NOT_NULL: "is not null"
        }
        
        if clause.cmp in [ComparisonOp.IS_NULL, ComparisonOp.NOT_NULL]:
            return f"{field} {op_map[clause.cmp]}"
        elif clause.cmp == ComparisonOp.CONTAINS:
            return f"{field} contains '{clause.value}'"
        elif clause.cmp == ComparisonOp.IN_RANGE:
            min_val, max_val = clause.value
            return f"({field} >= {min_val} && {field} <= {max_val})"
        else:
            return f"{field} {op_map[clause.cmp]} {clause.value}"

# LLM Function Definition for Query Tool
QUERY_CATALOGUE_TOOL = {
    "name": "query_catalogue",
    "description": """
    Query the ATNF Pulsar Catalogue using validated DSL.
    
    Returns: DataFrame with requested fields plus result provenance
    """,
    "input_schema": {
        "type": "object",
        "properties": {
            "query_dsl": {
                "type": "object",
                "description": "Query in validated DSL format"
            }
        },
        "required": ["query_dsl"]
    }
}
```

**Benefits of DSL Approach:**
- **Validation before execution**: Catch errors before hitting database
- **No LLM drift**: Operators are fixed enums, not free text
- **Composable**: Easy to build complex queries
- **Auditable**: Every query has a structured representation
- **Convertible**: Can translate to SQL, psrqpy, or native Python

**Example LLM Interaction:**

```
User: "Show me millisecond pulsars in globular clusters"

LLM generates:
{
  "tool": "query_catalogue",
  "query_dsl": {
    "select_fields": ["JNAME", "P0", "DM", "ASSOC"],
    "filters": {
      "op": "and",
      "clauses": [
        {"field": "P0", "cmp": "lt", "value": 0.03, "unit": "s"},
        {"field": "ASSOC", "cmp": "contains", "value": "GC"}
      ]
    },
    "order_by": "P0",
    "limit": 100
  }
}

Validator:
✓ All fields valid (JNAME, P0, DM, ASSOC exist in ATNF)
✓ P0 unit correct (seconds)
✓ Comparison operators valid
✓ Limit within bounds

Executor converts to psrqpy:
condition = "(P0 < 0.03 && ASSOC contains 'GC')"
```

---

## 3. Function/Tool Definitions

### 3.1 Core Query Functions

#### 3.1.1 `query_catalogue` (Enhanced)

```python
def query_catalogue(query_dsl: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Query ATNF catalogue with validated DSL
    
    Returns:
        results: DataFrame with requested fields
        provenance: Metadata about query execution
    """
    # Validate DSL
    try:
        validated_query = QueryDSL(**query_dsl)
    except ValidationError as e:
        return None, {
            'error': 'Query validation failed',
            'details': str(e),
            'suggestions': _suggest_corrections(query_dsl)
        }
    
    # Execute query
    catalogue = CatalogueInterface()
    results, execution_info = catalogue.query_pulsars_dsl(validated_query)
    
    # Build provenance
    provenance = {
        'catalogue_version': catalogue.catalogue_metadata['version'],
        'snapshot_date': catalogue.catalogue_metadata['snapshot_date'],
        'query_dsl': query_dsl,
        'condition_string': validated_query.to_psrqpy_condition(),
        'result_count': len(results),
        'fields_selected': validated_query.select_fields,
        'null_counts': {
            col: results[col].isna().sum() 
            for col in results.columns
        },
        'completeness_fraction': {
            col: 1.0 - (results[col].isna().sum() / len(results))
            for col in results.columns
        },
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    return results, provenance
```

**Key Enhancement: Result Provenance**

Every query result includes:
- Catalogue version and snapshot date
- Exact query DSL and condition string
- Number of matches
- Null counts for each field (critical for scientific safety)
- Data completeness fractions
- Timestamp for reproducibility

**LLM Response Template:**

```
I found {result_count} pulsars matching your criteria using ATNF catalogue 
version {version} (snapshot: {date}).

Results include:
- {field1}: {completeness}% measured
- {field2}: {completeness}% measured

Note: {null_count} pulsars in this selection have no measured {field}, which
may affect derived calculations.

[Display results table]

You can export this exact query as Python code or refine the search.
```

#### 3.1.2 `compute_derived_parameter` (ATNF-native preference)

```python
def compute_derived_parameter(
    pulsar_df: pd.DataFrame,
    parameter: str,
    use_atnf_native: bool = True
) -> Tuple[pd.Series, dict]:
    """
    Compute derived parameters, preferring ATNF-native when available
    
    Priority:
    1. Use ATNF-provided derived field if it exists (BSURF, EDOT, etc.)
    2. Compute using standard formulae if not in ATNF
    3. Record formula and assumptions
    """
    schema = SchemaGroundingPack()
    param_info = schema.canonical_params.get(parameter.upper())
    
    provenance = {
        'parameter': parameter,
        'source': None,
        'formula': None,
        'assumptions': {},
        'missing_count': 0
    }
    
    # Check if ATNF provides this derived parameter
    if use_atnf_native and parameter.upper() in pulsar_df.columns:
        provenance['source'] = 'atnf_native'
        provenance['formula'] = param_info.get('formula', 'ATNF-computed')
        result = pulsar_df[parameter.upper()]
    
    # Otherwise compute it
    else:
        provenance['source'] = 'computed'
        
        if parameter.upper() == 'BSURF':
            # Surface magnetic field
            # B_surf = 3.2e19 * sqrt(P * Pdot) Gauss
            P = pulsar_df['P0']
            Pdot = pulsar_df['P1']
            
            result = 3.2e19 * np.sqrt(P * Pdot)
            
            provenance['formula'] = 'B_surf = 3.2e19 * sqrt(P * P1) Gauss'
            provenance['assumptions'] = {
                'moment_of_inertia': '1.4e45 g cm^2',
                'neutron_star_radius': '10 km'
            }
        
        elif parameter.upper() == 'EDOT':
            # Spin-down luminosity
            # Edot = 4*pi^2 * I * F^3 * |Fdot|
            # or Edot = 4*pi^2 * I * Pdot / P^3
            
            if 'F0' in pulsar_df.columns and 'F1' in pulsar_df.columns:
                F = pulsar_df['F0']
                Fdot = pulsar_df['F1']
                I = 1.4e45  # g cm^2
                
                result = 4 * np.pi**2 * I * F**3 * np.abs(Fdot)
                provenance['formula'] = 'Edot = 4*pi^2 * I * F0^3 * |F1| erg/s'
            else:
                P = pulsar_df['P0']
                Pdot = pulsar_df['P1']
                I = 1.4e45
                
                result = 4 * np.pi**2 * I * Pdot / P**3
                provenance['formula'] = 'Edot = 4*pi^2 * I * P1 / P0^3 erg/s'
            
            provenance['assumptions'] = {
                'moment_of_inertia': '1.4e45 g cm^2'
            }
        
        elif parameter.upper() == 'CHAR_AGE':
            # Characteristic age (assuming constant Pdot)
            # tau = P / (2 * Pdot)
            P = pulsar_df['P0']
            Pdot = pulsar_df['P1']
            
            tau_seconds = P / (2 * Pdot)
            result = tau_seconds / (365.25 * 24 * 3600)  # Convert to years
            
            provenance['formula'] = 'tau = P0 / (2 * P1) [seconds -> years]'
            provenance['assumptions'] = {
                'braking_index': '3 (magnetic dipole)',
                'initial_period': 'P0 << P_current'
            }
        
        else:
            raise ValueError(f"Unknown derived parameter: {parameter}")
    
    # Count missing values
    provenance['missing_count'] = result.isna().sum()
    provenance['completeness'] = 1.0 - (provenance['missing_count'] / len(result))
    
    return result, provenance
```

**Key Principles:**

1. **ATNF-native first**: If ATNF catalogue includes BSURF, EDOT, etc., use those values
2. **Document everything**: Record exact formula and physical assumptions
3. **Track missingness**: Report how many pulsars have insufficient data
4. **Cite assumptions**: Moment of inertia, braking index, etc.

This keeps results consistent with published ATNF-based papers while maintaining transparency.

---

## 4. Scientific Safety Features

### 4.1 Null Handling and Data Quality

**Critical Implementation:**

```python
class ResultValidator:
    """Ensure scientific safety of query results"""
    
    def validate_result(self, df: pd.DataFrame, provenance: dict) -> dict:
        """
        Check result quality and generate warnings
        """
        warnings = []
        
        # Check for high missingness
        for field, null_count in provenance['null_counts'].items():
            if null_count > 0:
                frac_missing = null_count / provenance['result_count']
                if frac_missing > 0.5:
                    warnings.append({
                        'type': 'high_missingness',
                        'field': field,
                        'message': f"{field} is missing for {frac_missing*100:.0f}% of results. "
                                   f"Derived calculations may be incomplete."
                    })
        
        # Check for selection effects
        if 'ASSOC' in provenance.get('fields_selected', []):
            warnings.append({
                'type': 'selection_effect',
                'message': "Cluster associations are heterogeneous (different naming "
                           "conventions, historical classifications). Consider validating "
                           "matches manually for critical analyses."
            })
        
        # Check for epoch differences
        if 'POSEPOCH' in df.columns:
            epoch_range = df['POSEPOCH'].max() - df['POSEPOCH'].min()
            if epoch_range > 20:  # years
                warnings.append({
                    'type': 'epoch_range',
                    'message': f"Position epochs span {epoch_range:.0f} years. "
                               f"Proper motion may affect coordinates."
                })
        
        # Check result size
        if provenance['result_count'] == 0:
            warnings.append({
                'type': 'empty_result',
                'message': "No pulsars match your criteria. Consider relaxing filters.",
                'suggestions': self._suggest_relaxations(provenance['query_dsl'])
            })
        
        elif provenance['result_count'] > 1000:
            warnings.append({
                'type': 'large_result',
                'message': f"Query returned {provenance['result_count']} pulsars. "
                           f"Consider adding filters for more focused analysis."
            })
        
        return {
            'is_safe': len([w for w in warnings if w['type'] in ['high_missingness']]) == 0,
            'warnings': warnings
        }
    
    def _suggest_relaxations(self, query_dsl):
        """Suggest how to broaden an overly restrictive query"""
        suggestions = []
        
        if query_dsl.get('filters'):
            for clause in query_dsl['filters'].get('clauses', []):
                if clause['cmp'] in ['lt', 'le']:
                    suggestions.append(f"Increase {clause['field']} threshold (currently < {clause['value']})")
                elif clause['cmp'] in ['gt', 'ge']:
                    suggestions.append(f"Decrease {clause['field']} threshold (currently > {clause['value']})")
                elif clause['cmp'] == 'contains':
                    suggestions.append(f"Try broader association match (currently '{clause['value']}')")
        
        return suggestions
```

### 4.2 "Show Query Plan" Feature

**Implementation:**

```python
def generate_query_plan(query_dsl: dict) -> str:
    """
    Generate human-readable explanation of query execution
    """
    plan_parts = []
    
    validated_query = QueryDSL(**query_dsl)
    
    # Starting point
    plan_parts.append("1. Load ATNF catalogue (version: runtime-detected)")
    
    # Filters
    if validated_query.filters:
        plan_parts.append("2. Apply filters:")
        for clause in validated_query.filters.clauses:
            plan_parts.append(f"   - {clause.field} {clause.cmp} {clause.value}")
    
    # Selection
    if validated_query.select_fields:
        plan_parts.append(f"3. Select fields: {', '.join(validated_query.select_fields)}")
    else:
        plan_parts.append("3. Select all available fields")
    
    # Ordering
    if validated_query.order_by:
        plan_parts.append(f"4. Sort by {validated_query.order_by}")
    
    # Limit
    if validated_query.limit:
        plan_parts.append(f"5. Limit to first {validated_query.limit} results")
    
    # Add psrqpy condition
    condition = validated_query.to_psrqpy_condition()
    plan_parts.append(f"\nPsrqpy condition string: {condition}")
    
    # Add reproducible Python code
    plan_parts.append("\n--- Reproducible Python Code ---")
    plan_parts.append("from psrqpy import QueryATNF")
    plan_parts.append(f"q = QueryATNF(")
    plan_parts.append(f"    params={validated_query.select_fields},")
    plan_parts.append(f"    condition='{condition}'")
    plan_parts.append(")")
    plan_parts.append("results = q.pandas")
    
    return "\n".join(plan_parts)
```

**User Interface:**

```
User: "Show me millisecond pulsars in globular clusters"
[Assistant shows results table]

User: "Can you show me the query plan for that?"

[Assistant generates query plan:]

Query Execution Plan:
1. Load ATNF catalogue (version: runtime-detected)
2. Apply filters:
   - P0 lt 0.03 (seconds)
   - ASSOC contains GC
3. Select fields: JNAME, P0, DM, ASSOC
4. Sort by P0
5. Limit to first 100 results

Psrqpy condition string: (P0 < 0.03 && ASSOC contains 'GC')

--- Reproducible Python Code ---
from psrqpy import QueryATNF
q = QueryATNF(
    params=['JNAME', 'P0', 'DM', 'ASSOC'],
    condition='(P0 < 0.03 && ASSOC contains \'GC\')'
)
results = q.pandas
```

---

## 5. Analysis Tools (Unchanged from Original)

### 5.1 Statistical Analysis

```python
def statistical_analysis(df: pd.DataFrame, parameters: List[str]) -> dict:
    """
    Compute statistical summaries for specified parameters
    
    Returns:
        summary: Dictionary with stats for each parameter
    """
    summary = {}
    
    for param in parameters:
        if param not in df.columns:
            continue
            
        data = df[param].dropna()
        
        summary[param] = {
            'count': len(data),
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'quartiles': {
                'q25': float(data.quantile(0.25)),
                'q75': float(data.quantile(0.75))
            }
        }
    
    return summary
```

### 5.2 Correlation Analysis

```python
def correlation_analysis(
    df: pd.DataFrame,
    param_x: str,
    param_y: str
) -> dict:
    """
    Compute correlation between two parameters
    """
    from scipy.stats import pearsonr, spearmanr
    
    # Remove rows with missing data
    clean_df = df[[param_x, param_y]].dropna()
    
    if len(clean_df) < 3:
        return {'error': 'Insufficient data for correlation'}
    
    x = clean_df[param_x]
    y = clean_df[param_y]
    
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    
    return {
        'n_points': len(clean_df),
        'pearson': {
            'correlation': float(pearson_r),
            'p_value': float(pearson_p)
        },
        'spearman': {
            'correlation': float(spearman_r),
            'p_value': float(spearman_p)
        }
    }
```

---

## 6. Visualization Tools (Unchanged from Original)

### 6.1 Period-Period Derivative Diagram

```python
def create_pp_diagram(
    df: pd.DataFrame,
    highlight_groups: Optional[dict] = None
) -> plotly.graph_objects.Figure:
    """
    Create P-Pdot diagram with classification regions
    
    Args:
        df: DataFrame with P0 and P1 columns
        highlight_groups: Dict mapping group names to filter conditions
    """
    import plotly.graph_objects as go
    import numpy as np
    
    fig = go.Figure()
    
    # Add classification region boundaries
    P_range = np.logspace(-3, 1, 100)
    
    # Lines of constant age
    for age_yr in [1e3, 1e6, 1e9]:
        Pdot = P_range / (2 * age_yr * 365.25 * 24 * 3600)
        fig.add_trace(go.Scatter(
            x=P_range,
            y=Pdot,
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name=f'Age = {age_yr:.0e} yr',
            showlegend=True
        ))
    
    # Lines of constant magnetic field
    for B_gauss in [1e8, 1e10, 1e12, 1e14]:
        Pdot = (B_gauss / 3.2e19)**2 / P_range
        fig.add_trace(go.Scatter(
            x=P_range,
            y=Pdot,
            mode='lines',
            line=dict(color='lightblue', dash='dot'),
            name=f'B = {B_gauss:.0e} G',
            showlegend=True
        ))
    
    # Plot pulsars
    if highlight_groups:
        for group_name, condition in highlight_groups.items():
            group_df = df.query(condition) if condition else df
            fig.add_trace(go.Scatter(
                x=group_df['P0'],
                y=group_df['P1'],
                mode='markers',
                name=group_name,
                text=group_df['JNAME'],
                hovertemplate='<b>%{text}</b><br>P=%{x:.4f} s<br>Pdot=%{y:.2e}<extra></extra>'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df['P0'],
            y=df['P1'],
            mode='markers',
            name='Pulsars',
            text=df['JNAME'],
            hovertemplate='<b>%{text}</b><br>P=%{x:.4f} s<br>Pdot=%{y:.2e}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Period - Period Derivative Diagram',
        xaxis=dict(title='Period (s)', type='log'),
        yaxis=dict(title='Period Derivative', type='log'),
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig
```

---

## 7. LLM System Prompt (Updated)

### 7.1 Enhanced System Prompt

```
You are an expert astronomy assistant specializing in pulsar science and the ATNF 
Pulsar Catalogue. You help researchers query pulsar data, generate visualizations, 
and perform statistical analyses through natural language.

## Core Principles

1. **Use tools for truth**: Never guess parameter values or pulsar properties. Always 
   use query_catalogue to retrieve factual data.

2. **Validate before executing**: All queries must use the validated DSL format. 
   Never use free-form filter strings.

3. **Prefer ATNF-native derived fields**: When computing derived parameters like BSURF 
   or EDOT, check if ATNF provides them first. If you compute them yourself, always 
   state the formula and assumptions.

4. **Report catalogue metadata**: Include catalogue version and snapshot date in responses 
   when presenting query results. Never hardcode version numbers.

5. **Handle missingness explicitly**: Always report how many pulsars have measured values 
   for requested parameters. Warn when missingness exceeds 50%.

6. **Provide provenance**: Include information about query execution, data completeness, 
   and any assumptions made in derivations.

## Catalogue Knowledge

The ATNF Pulsar Catalogue contains:
- Dynamically determined pulsar count (query at runtime)
- Version information detected from local cache
- 100+ parameters including timing, astrometric, binary, and derived quantities

### Parameter Naming Conventions

Canonical ATNF parameter codes (use these in queries):
- Period: P0 (seconds)
- Period derivative: P1 (dimensionless) - NOTE: "Pdot" is an alias, not canonical
- Frequency: F0 (Hz)
- Frequency derivative: F1 (s^-2)
- Dispersion measure: DM (pc cm^-3)
- Surface magnetic field: BSURF (Gauss) - ATNF-provided when available
- Spin-down luminosity: EDOT (erg/s) - ATNF-provided when available

Natural language aliases:
- "period" → P0
- "spin-down rate" or "Pdot" → P1
- "magnetic field" → BSURF
- "millisecond pulsar" → P0 < 0.030 s
- "globular cluster" → ASSOC contains relevant cluster name

## Query DSL Format

ALL queries must use this validated structure:

{
  "select_fields": ["JNAME", "P0", "DM"],
  "filters": {
    "op": "and",  // or "or"
    "clauses": [
      {
        "field": "P0",
        "cmp": "lt",  // eq, ne, lt, le, gt, ge, contains, in_range, is_null, not_null
        "value": 0.03,
        "unit": "s"
      }
    ]
  },
  "order_by": "P0",
  "limit": 100
}

NEVER use free-form condition strings like:
- WRONG: {'P0': '<0.030'}
- CORRECT: {"field": "P0", "cmp": "lt", "value": 0.03}

## Response Guidelines

When presenting query results:

1. State catalogue version and snapshot date
2. Report number of matches
3. Note any high missingness (>50%) in requested fields
4. Highlight any selection effects or caveats
5. Offer to show query plan or export code

Example good response:
"I found 127 binary millisecond pulsars using ATNF catalogue version 2.6.4 
(snapshot: 2025-08-19). Results include:
- P0: 100% measured
- DM: 98% measured  
- PB: 100% measured (required for binary selection)

Note: 15 pulsars have no measured eccentricity (ECC), which may limit some 
binary evolution analyses.

[show results table]

Would you like me to:
1. Show the query plan and reproducible code?
2. Create a P-Pdot diagram?
3. Compute derived parameters?"

## Scientific Safety

- Always validate filter values against typical ranges
- Warn about empty results and suggest relaxations
- Note when associations (ASSOC) use heterogeneous naming
- Check for epoch differences in timing parameters
- Report completeness fractions for derived calculations

## Available Tools

Use these tools to answer queries:
- query_catalogue: Execute validated DSL queries
- compute_derived_parameter: Calculate physics quantities (prefer ATNF-native)
- statistical_analysis: Compute summary statistics
- correlation_analysis: Analyze relationships between parameters
- create_visualization: Generate plots (P-Pdot diagrams, histograms, etc.)
- generate_query_plan: Show execution plan and reproducible code

## When You Don't Know

If uncertain about:
- Parameter naming → Ask user or search schema
- Filter values → Use query_catalogue to explore ranges first
- Physical interpretation → State "based on standard pulsar physics" and cite assumptions

Never hallucinate data. Always query the catalogue.
```

---

## 8. Testing Strategy (Enhanced)

### 8.1 Unit Tests (Unchanged from Original)

```python
import pytest
from atnf_interface import CatalogueInterface, QueryDSL, compute_derived_parameter

def test_query_dsl_validation():
    """Test DSL validation catches errors"""
    
    # Valid query
    valid_dsl = {
        "select_fields": ["JNAME", "P0"],
        "filters": {
            "op": "and",
            "clauses": [
                {"field": "P0", "cmp": "lt", "value": 0.03}
            ]
        }
    }
    query = QueryDSL(**valid_dsl)
    assert query.to_psrqpy_condition() == "(P0 < 0.03)"
    
    # Invalid field
    invalid_dsl = {
        "select_fields": ["INVALID_FIELD"],
        "filters": None
    }
    with pytest.raises(ValueError):
        QueryDSL(**invalid_dsl)

def test_derived_parameter_atnf_native():
    """Test preference for ATNF-native derived fields"""
    catalogue = CatalogueInterface()
    
    # Get some pulsars
    query_dsl = {
        "select_fields": ["JNAME", "P0", "P1", "BSURF"],
        "filters": {
            "op": "and",
            "clauses": [
                {"field": "BSURF", "cmp": "not_null"}
            ]
        },
        "limit": 10
    }
    results, prov = catalogue.query_pulsars_dsl(QueryDSL(**query_dsl))
    
    # Compute BSURF (should use ATNF-native)
    bsurf_computed, prov = compute_derived_parameter(
        results,
        'BSURF',
        use_atnf_native=True
    )
    
    assert prov['source'] == 'atnf_native'
    assert prov['formula'] == 'ATNF-computed'

def test_null_handling():
    """Test that null values are properly tracked"""
    catalogue = CatalogueInterface()
    
    query_dsl = {
        "select_fields": ["JNAME", "P0", "ECC"],
        "filters": {
            "op": "and",
            "clauses": [
                {"field": "P0", "cmp": "lt", "value": 0.03}
            ]
        },
        "limit": 100
    }
    
    results, prov = catalogue.query_pulsars_dsl(QueryDSL(**query_dsl))
    
    # Check provenance includes null counts
    assert 'null_counts' in prov
    assert 'completeness_fraction' in prov
    assert 'ECC' in prov['null_counts']
    
    # High missingness should be flagged
    from atnf_interface import ResultValidator
    validator = ResultValidator()
    validation = validator.validate_result(results, prov)
    
    if prov['null_counts']['ECC'] > len(results) * 0.5:
        assert any(w['type'] == 'high_missingness' for w in validation['warnings'])
```

### 8.2 Integration Tests (Unchanged from Original)

```python
def test_end_to_end_query_flow():
    """Test complete query execution with LLM"""
    
    # Simulate LLM query
    user_query = "Show me millisecond pulsars with P < 10 ms"
    
    # LLM should generate this DSL
    llm_generated_dsl = {
        "select_fields": ["JNAME", "P0", "DM"],
        "filters": {
            "op": "and",
            "clauses": [
                {"field": "P0", "cmp": "lt", "value": 0.01, "unit": "s"}
            ]
        },
        "order_by": "P0",
        "limit": 100
    }
    
    # Execute query
    catalogue = CatalogueInterface()
    results, prov = catalogue.query_pulsars_dsl(QueryDSL(**llm_generated_dsl))
    
    # Verify results
    assert len(results) > 0
    assert all(results['P0'] < 0.01)
    assert 'catalogue_version' in prov
    assert 'snapshot_date' in prov
```

### 8.3 LLM Accuracy Benchmarks (Enhanced)

```python
# benchmarks/pulsar_queries.json
{
  "test_cases": [
    {
      "query": "What is the period of the Vela pulsar?",
      "expected_parameter": "P0",
      "expected_pulsar": "B0833-45",
      "expected_answer_contains": "0.089",
      "difficulty": "easy"
    },
    {
      "query": "How many millisecond pulsars are in 47 Tucanae?",
      "expected_tool_calls": ["query_catalogue"],
      "expected_dsl": {
        "filters": {
          "op": "and",
          "clauses": [
            {"field": "P0", "cmp": "lt", "value": 0.03},
            {"field": "ASSOC", "cmp": "contains", "value": "47 Tuc"}
          ]
        }
      },
      "expected_answer_type": "integer",
      "difficulty": "medium"
    },
    {
      "query": "Compare the magnetic field distribution of recycled vs normal pulsars",
      "expected_tool_calls": ["query_catalogue", "compute_derived_parameter", 
                            "statistical_analysis", "create_visualization"],
      "expected_parameters": ["P0", "P1"],
      "expected_derived": "BSURF",
      "expected_provenance": {
        "source": "atnf_native",  # Should prefer ATNF's BSURF
        "formula_documented": true
      },
      "difficulty": "hard"
    },
    {
      "query": "Show me pulsars with periods around 1.4 ms",
      "expected_dsl": {
        "filters": {
          "op": "and",
          "clauses": [
            {"field": "P0", "cmp": "in_range", "value": [0.0013, 0.0015]}
          ]
        }
      },
      "difficulty": "medium"
    },
    {
      "query": "Find all pulsars discovered in the last 2 years",
      "expected_response_type": "clarifying_question",
      "expected_clarification": "Catalogue doesn't track discovery date explicitly. Could search by reference epoch or suggest alternative approaches.",
      "difficulty": "medium"
    }
  ],
  
  "failure_case_tests": [
    {
      "query": "Show me pulsars with spin-down ages",
      "test_type": "ambiguous_term",
      "expected_behavior": "clarifying_question",
      "expected_clarification": "Do you want characteristic age (P/2Pdot) or true age? Note assumptions for each.",
      "difficulty": "medium"
    },
    {
      "query": "Plot DM vs period for all pulsars with measured braking index",
      "test_type": "missingness_handling",
      "expected_behavior": "execute_with_warning",
      "expected_warning": "Braking index (n) is measured for only ~10 pulsars. Result will be sparse.",
      "expected_tool_calls": ["query_catalogue", "create_visualization"],
      "difficulty": "hard"
    },
    {
      "query": "Show me all pulsars with orbital periods less than 1 hour",
      "test_type": "empty_result",
      "expected_behavior": "graceful_failure",
      "expected_response_contains": "No pulsars match",
      "expected_suggestions": [
        "Relax orbital period threshold",
        "Shortest known binary period is ~90 minutes"
      ],
      "difficulty": "easy"
    },
    {
      "query": "Give me everything about J0737-3039",
      "test_type": "overly_broad",
      "expected_behavior": "focused_response",
      "expected_response_contains": [
        "Double pulsar system",
        "specific parameters",
        "offer to focus on particular aspects"
      ],
      "difficulty": "medium"
    },
    {
      "query": "Find energetic pulsars",
      "test_type": "vague_term",
      "expected_behavior": "clarifying_question",
      "expected_clarification": "By 'energetic' do you mean: high spin-down luminosity (Edot), high magnetic field, or young age?",
      "difficulty": "medium"
    }
  ]
}
```

**Evaluation Metrics (Enhanced):**
- **Tool Call Accuracy**: Did LLM call correct functions?
- **DSL Format Compliance**: Used validated DSL instead of free-form?
- **Parameter Mapping Accuracy**: Correct ATNF parameter codes?
- **ATNF-Native Preference**: Used ATNF-derived fields when available?
- **Provenance Completeness**: Included version, null counts, assumptions?
- **Response Relevance**: Answer addresses question?
- **Scientific Accuracy**: Correct physical interpretation?
- **Graceful Failure**: Appropriate handling of ambiguous/impossible queries?

```python
# benchmarks/evaluate.py
class BenchmarkEvaluator:
    def evaluate_test_case(self, test_case, response):
        scores = {
            'tool_call_accuracy': self._check_tool_calls(
                test_case.get('expected_tool_calls'),
                response.tool_calls
            ),
            'dsl_compliance': self._check_dsl_format(
                response.query_dsl
            ),
            'parameter_accuracy': self._check_parameters(
                test_case.get('expected_parameters'),
                response.parameters_used
            ),
            'provenance_completeness': self._check_provenance(
                test_case.get('expected_provenance'),
                response.provenance
            ),
            'filter_accuracy': self._check_filters(
                test_case.get('expected_dsl'),
                response.query_dsl
            ),
            'answer_correctness': self._check_answer(
                test_case.get('expected_answer_contains'),
                response.answer_text
            ),
            'failure_handling': self._check_failure_handling(
                test_case,
                response
            )
        }
        return scores
    
    def _check_dsl_format(self, query_dsl):
        """Verify query uses validated DSL, not free-form filters"""
        if query_dsl is None:
            return 1.0  # No query needed
        
        try:
            QueryDSL(**query_dsl)
            return 1.0
        except:
            return 0.0
    
    def _check_provenance(self, expected_prov, actual_prov):
        """Check if response includes required provenance"""
        if expected_prov is None:
            return 1.0
        
        score = 0
        checks = [
            'catalogue_version' in actual_prov,
            'snapshot_date' in actual_prov,
            'null_counts' in actual_prov,
            'result_count' in actual_prov
        ]
        
        if expected_prov.get('source') == 'atnf_native':
            checks.append(actual_prov.get('source') == 'atnf_native')
        
        return sum(checks) / len(checks)
    
    def _check_failure_handling(self, test_case, response):
        """Check handling of failure cases"""
        if test_case.get('test_type') == 'ambiguous_term':
            return 1.0 if response.asks_clarification else 0.0
        elif test_case.get('test_type') == 'empty_result':
            has_suggestion = any('suggest' in k.lower() for k in response.keys())
            return 1.0 if has_suggestion else 0.0
        elif test_case.get('test_type') == 'missingness_handling':
            has_warning = any('warning' in k.lower() for k in response.keys())
            return 1.0 if has_warning else 0.0
        else:
            return 1.0  # Not a failure case
    
    def run_full_benchmark(self):
        """Run all test cases and report aggregate metrics"""
        results = []
        for test_case in self.load_test_cases():
            response = self.chat_interface.process_query(
                test_case['query']
            )
            scores = self.evaluate_test_case(test_case, response)
            results.append({
                'query': test_case['query'],
                'difficulty': test_case['difficulty'],
                'test_type': test_case.get('test_type', 'standard'),
                **scores
            })
        
        # Aggregate by difficulty and test type
        import pandas as pd
        df = pd.DataFrame(results)
        
        return {
            'by_difficulty': df.groupby('difficulty').mean(),
            'by_test_type': df.groupby('test_type').mean(),
            'overall': df.mean()
        }
```

**Target Performance:**
- Easy queries: >95% accuracy
- Medium queries: >85% accuracy  
- Hard queries: >70% accuracy
- Failure case handling: >90% appropriate response

---

## 9. User Experience Design (Unchanged from Original)

### 9.1 Example Conversation Flows

**Flow 1: Exploratory Research**
```
User: Hi, I'm interested in binary millisecond pulsars
Assistant: Great! Binary millisecond pulsars are fascinating systems. Let me show you 
what we have in the catalogue.

[Calls: query_catalogue with DSL for BINARY=1 and P0<0.030]

I found 127 binary millisecond pulsars using ATNF catalogue version 2.6.4 
(snapshot: 2025-08-19). Results include:
- P0: 100% measured (required for MSP selection)
- DM: 98% measured  
- PB (orbital period): 100% measured (required for binary selection)
- ECC (eccentricity): 88% measured

Note: 15 pulsars have no measured eccentricity, which may limit some binary 
evolution analyses.

Some highlights:
- Shortest period: J1748-2446ad (1.4 ms, in a globular cluster)
- Most eccentric MSP binary: J1903+0327 (e=0.44)
- Shortest orbital period: J1311-3430 (94 minutes)

Would you like me to:
1. Create a plot of orbital period vs pulsar spin period?
2. Show the distribution of eccentricities?
3. Filter for specific orbital period ranges?

User: Yes, show me the orbital period vs spin period plot

[Assistant creates visualization and analyzes]

Yes! There is a clear pattern. Most MSP binaries with orbital periods < 10 days have 
near-circular orbits (ECC < 0.0001). This is because tidal circularization during the 
mass transfer phase removed eccentricity. 

However, pulsars in wider orbits (PB > 50 days) can have significant eccentricity. 
These likely formed through different channels - possibly exchange interactions in 
globular clusters or triple systems.

Want to explore more? I can:
- Export this data and visualization code
- Show you specific examples of high-eccentricity MSP binaries
- Compare with normal pulsar binaries
```

This conversation demonstrates:
- Natural exploration with provenance
- Explicit data quality reporting
- Offering relevant follow-up options
- Providing scientific context
- Maintaining conversation flow

---

## 10. Conclusion & Implementation Recommendation

### 10.1 Executive Summary

This technical brief outlines a comprehensive plan for creating an LLM-powered natural language interface to the ATNF Pulsar Catalogue. The proposed system will transform how astronomers interact with pulsar data by enabling conversational queries, automated visualizations, and intelligent analysis suggestions - all with strong scientific safety guarantees.

### 10.2 Core Design Decisions

**Architecture**: Web-based chat interface with FastAPI backend  
**LLM**: Anthropic Claude Sonnet 4.5 (primary) with local fallback option  
**Data Layer**: psrqpy integration with versioned snapshots  
**Query Language**: Validated DSL with comprehensive error handling  
**Deployment**: Cloud platform for public access, containerized for institutional deployment

### 10.3 Key Innovations

1. **Validated Query DSL**: Eliminates LLM drift and enables pre-execution validation
2. **Result Provenance**: Every query includes catalogue version, null counts, and completeness metrics
3. **ATNF-Native Preference**: Uses official derived parameters when available, documents custom computations
4. **Scientific Safety**: Explicit handling of missingness, selection effects, and data quality
5. **Automated Schema Grounding**: Regenerates parameter mappings from ATNF documentation
6. **Reproducibility**: All queries exportable as validated Python code

### 10.4 Implementation Priorities

**Phase 1 (Weeks 1-3)**: Core infrastructure
- Implement QueryDSL validation layer
- Build CatalogueInterface with versioning
- Create automated schema grounding pack
- Deploy basic query execution with provenance

**Phase 2 (Weeks 4-5)**: LLM integration and tools
- Integrate Claude API with function calling
- Implement compute_derived_parameter with ATNF preference
- Add result validation and warning system
- Build query plan generation

**Phase 3 (Weeks 6-8)**: Visualization, testing, and deployment
- Create visualization tools (P-Pdot diagrams, etc.)
- Implement comprehensive benchmark suite (including failure cases)
- Build web interface with code export
- Deploy and gather community feedback

**Estimated Total Development Time**: 2-3 months  
**Estimated Operational Costs**: $20-50/month after optimization

### 10.5 Success Criteria

- Query accuracy >90% on standard benchmark suite
- Failure case handling >90% (graceful degradation)
- DSL format compliance >95%
- Response time <3 seconds average
- All results include provenance metadata
- User satisfaction from astronomy community feedback

### 10.6 Risk Mitigation

Primary risks (LLM hallucination, API costs, scientific accuracy) are addressed through:
- **Validated DSL**: Prevents malformed queries before execution
- **Provenance tracking**: Ensures all results are auditable
- **ATNF-native preference**: Maintains consistency with published literature
- **Null handling**: Explicit reporting of data quality and completeness
- **Comprehensive benchmarking**: Including failure cases and edge conditions
- **Caching strategies**: Reduce API costs
- **Code export**: Transparent tool calling enables verification

### 10.7 Comparison to Original Brief

**Version 1.1 Key Improvements:**
1. **No hardcoded catalogue versions** - runtime detection and reporting
2. **Strict query DSL** - replaces free-form filter strings
3. **ATNF-native preference** - prioritizes official derived parameters
4. **Enhanced provenance** - tracks version, nulls, completeness, assumptions
5. **Automated schema grounding** - regenerates from ATNF docs
6. **Expanded benchmarks** - includes failure cases and ambiguous queries
7. **Scientific safety features** - result validation, warnings, suggestions

### 10.8 Long-Term Vision

This project establishes a template for LLM interfaces to astronomical catalogues more broadly. Future extensions could include:
- Multi-catalogue integration (Fermi, INTEGRAL, Chandra)
- Cross-matching capabilities with proper uncertainty propagation
- Literature search and citation of relevant papers
- Automated observation planning and proposal generation
- Integration with pulsar timing software (TEMPO2, PINT)

### 10.9 Community Impact

By lowering the barrier to catalogue access while maintaining scientific rigor, this tool will:
- Enable faster exploratory research with quality guarantees
- Facilitate astronomy education with transparent provenance
- Improve reproducibility through validated code export
- Foster new research questions through AI-assisted exploration
- Maintain consistency with ATNF-based literature through native parameter preference

### 10.10 Final Recommendation

**Proceed with implementation** using the enhanced architecture outlined in this brief. The combination of:
- Mature libraries (psrqpy, FastAPI)
- Powerful LLMs (Claude) with validated tool contracts
- Strong provenance and safety guarantees
- Clear user need and community benefit

makes this a high-value, achievable project with robust scientific foundations.

The pulsar astronomy community will benefit significantly from a tool that combines the precision of traditional querying with the flexibility of natural language interaction - without sacrificing scientific accuracy or reproducibility.

---

**Next Steps:**
1. Set up development environment and repository
2. Implement QueryDSL validation layer with pydantic
3. Build CatalogueInterface with versioning and provenance
4. Create automated schema grounding pack from ATNF docs
5. Integrate Claude API with validated function calling
6. Build comprehensive benchmark suite (standard + failure cases)
7. Develop simple web interface for testing
8. Gather feedback from astronomy colleagues
9. Iterate based on real-world usage patterns

---

*This technical brief (v1.1) provides a comprehensive roadmap for building a next-generation interface to astronomical catalogue data with strong scientific safety guarantees. The design is practical, cost-effective, aligned with modern research workflows, and maintains the scientific rigor required for publishable research.*

## Appendix A: Quick Reference - Key Changes from v1.0

1. **Dynamic Catalogue Versioning** (Section 1.1, 2.3.1)
   - Removed hardcoded "version 2.7.0, ~3000 pulsars"
   - Runtime detection from psrqpy metadata
   - Snapshot dating for reproducibility

2. **Validated Query DSL** (Section 2.3.3, 3.1.1)
   - Replaced free-form filters with pydantic-validated DSL
   - Enum-based comparison operators
   - Pre-execution validation

3. **ATNF-Native Preference** (Section 3.1.2)
   - Prioritize BSURF, EDOT from ATNF when available
   - Document formulas and assumptions for custom computations
   - Maintain consistency with published literature

4. **Enhanced Provenance** (Section 3.1.1, 4.1)
   - Catalogue version and snapshot date
   - Null counts and completeness fractions
   - Result validation with warnings

5. **Automated Schema Grounding** (Section 2.3.2)
   - Generate mappings from ATNF documentation
   - Separate canonical codes from human aliases
   - Regenerate when ATNF adds parameters

6. **Expanded Benchmarks** (Section 8.3)
   - Added failure case tests
   - Ambiguous term handling
   - Empty result graceful degradation
   - Missingness warnings

7. **Scientific Safety Features** (Section 4)
   - ResultValidator class
   - Query plan generation
   - Suggestion system for failed queries

## Appendix B: External Review Integration

This version incorporates feedback from external technical review focusing on:
- Robust scientific safety practices
- Maintainable architecture
- Consistency with ATNF conventions
- Prevention of common LLM interface failure modes

All suggestions have been evaluated and integrated where appropriate to strengthen the technical foundation while maintaining practical implementability.

---