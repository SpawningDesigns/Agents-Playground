#!/usr/bin/env python
import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import find_dotenv, load_dotenv

from draftr.crew import Draftr

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Walk up from the current working directory to find the nearest .env
# (typically Agents-Playground/.env when draftr runs as a workspace member).
load_dotenv(find_dotenv())

_PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "output"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(config_path: Path = DEFAULT_CONFIG) -> None:
    """Configure root logger. Level is read from config.yaml logging.level (default INFO)."""
    cfg: dict = {}
    if config_path.exists():
        with config_path.open() as fh:
            cfg = yaml.safe_load(fh) or {}
    level_str = cfg.get("logging", {}).get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    rfp_path: Path
    client_name: str
    rfp_tone: str
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)


def load_config(config_path: Path) -> dict:
    """Load config.yaml; return empty dict if the file doesn't exist."""
    if config_path.exists():
        with config_path.open() as fh:
            return yaml.safe_load(fh) or {}
    return {}


def resolve_inputs(
    rfp_path: Optional[str],
    client_name: Optional[str],
    rfp_tone: Optional[str],
    output_dir: Optional[str],
    config_path: Path = DEFAULT_CONFIG,
) -> RunConfig:
    """
    Merge runtime values with config.yaml.
    Priority: CLI / trigger args > config.yaml > built-in defaults.
    Raises on missing or invalid required fields.
    """
    cfg = load_config(config_path)
    rfp_cfg = cfg.get("rfp", {})
    out_cfg = cfg.get("output", {})

    resolved_path_str = rfp_path or rfp_cfg.get("path") or ""
    resolved_client = client_name or rfp_cfg.get("client_name") or ""
    resolved_tone = rfp_tone or rfp_cfg.get("tone") or "formal government"
    resolved_output = Path(output_dir or out_cfg.get("dir") or DEFAULT_OUTPUT_DIR)

    if not resolved_path_str:
        raise ValueError(
            "rfp_path is required. Supply it via --rfp-path or set rfp.path in config.yaml."
        )
    resolved_rfp = Path(resolved_path_str)
    if not resolved_rfp.is_file():
        raise FileNotFoundError(
            f"RFP file not found: '{resolved_rfp}'. "
            "Check the path or update rfp.path in config.yaml."
        )
    if not resolved_client:
        raise ValueError(
            "client_name is required. Supply it via --client-name or set rfp.client_name in config.yaml."
        )

    return RunConfig(
        rfp_path=resolved_rfp.resolve(),
        client_name=resolved_client,
        rfp_tone=resolved_tone,
        output_dir=resolved_output.resolve(),
    )


def build_crew_inputs(run_config: RunConfig) -> dict:
    """Convert a RunConfig to the flat dict CrewAI passes into task templates."""
    run_config.output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "rfp_path": str(run_config.rfp_path),
        "client_name": run_config.client_name,
        "rfp_tone": run_config.rfp_tone,
        "output_dir": str(run_config.output_dir),
    }


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------

def _base_parser(description: str) -> argparse.ArgumentParser:
    """Shared RFP flags used by run, train, and test entry points."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--rfp-path", metavar="PATH", help="Path to the RFP PDF file")
    parser.add_argument("--client-name", metavar="NAME", help="Client / organisation name")
    parser.add_argument(
        "--rfp-tone",
        metavar="TONE",
        help="Proposal tone, e.g. 'formal government' or 'commercial enterprise'",
    )
    parser.add_argument("--output-dir", metavar="DIR", help="Directory for output files")
    parser.add_argument(
        "--config",
        metavar="FILE",
        default=str(DEFAULT_CONFIG),
        help=f"Path to config.yaml (default: {DEFAULT_CONFIG})",
    )
    return parser


def _run_config_from_args(args: argparse.Namespace) -> RunConfig:
    return resolve_inputs(
        rfp_path=args.rfp_path,
        client_name=args.client_name,
        rfp_tone=args.rfp_tone,
        output_dir=args.output_dir,
        config_path=Path(args.config),
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run():
    """Run the Draftr crew end-to-end."""
    parser = _base_parser("Run the Draftr RFP-response crew.")
    args = parser.parse_args()
    setup_logging(Path(args.config))

    try:
        run_config = _run_config_from_args(args)
    except (ValueError, FileNotFoundError) as e:
        log.error("%s", e)
        sys.exit(1)

    log.info("Starting crew — client: %s | RFP: %s", run_config.client_name, run_config.rfp_path.name)
    try:
        Draftr().crew().kickoff(inputs=build_crew_inputs(run_config))
        log.info("Crew finished. Outputs in: %s", run_config.output_dir)
    except Exception as e:
        log.error("Crew failed: %s", e, exc_info=True)
        raise


def train():
    """Train the crew for a given number of iterations."""
    parser = _base_parser("Train the Draftr crew.")
    parser.add_argument("n_iterations", type=int, help="Number of training iterations")
    parser.add_argument("filename", help="File to save training results")
    args = parser.parse_args()
    setup_logging(Path(args.config))

    try:
        run_config = _run_config_from_args(args)
    except (ValueError, FileNotFoundError) as e:
        log.error("%s", e)
        sys.exit(1)

    log.info("Training crew — %d iteration(s) → %s", args.n_iterations, args.filename)
    try:
        Draftr().crew().train(
            n_iterations=args.n_iterations,
            filename=args.filename,
            inputs=build_crew_inputs(run_config),
        )
        log.info("Training complete.")
    except Exception as e:
        log.error("Training failed: %s", e, exc_info=True)
        raise


def replay():
    """Replay crew execution from a specific task ID."""
    parser = argparse.ArgumentParser(description="Replay Draftr crew from a specific task.")
    parser.add_argument("task_id", help="ID of the task to replay from")
    parser.add_argument("--config", metavar="FILE", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    setup_logging(Path(args.config))

    log.info("Replaying from task: %s", args.task_id)
    try:
        Draftr().crew().replay(task_id=args.task_id)
        log.info("Replay complete.")
    except Exception as e:
        log.error("Replay failed: %s", e, exc_info=True)
        raise


def test():
    """Test the crew and return evaluation results."""
    parser = _base_parser("Test the Draftr crew.")
    parser.add_argument("n_iterations", type=int, help="Number of test iterations")
    parser.add_argument("eval_llm", help="LLM model ID to use for evaluation")
    args = parser.parse_args()
    setup_logging(Path(args.config))

    try:
        run_config = _run_config_from_args(args)
    except (ValueError, FileNotFoundError) as e:
        log.error("%s", e)
        sys.exit(1)

    log.info("Testing crew — %d iteration(s), evaluator: %s", args.n_iterations, args.eval_llm)
    try:
        Draftr().crew().test(
            n_iterations=args.n_iterations,
            eval_llm=args.eval_llm,
            inputs=build_crew_inputs(run_config),
        )
        log.info("Test complete.")
    except Exception as e:
        log.error("Test failed: %s", e, exc_info=True)
        raise


def run_with_trigger():
    """Run the crew from a JSON trigger payload (used by remote schedulers).

    Accepts the payload from argv[1] or piped stdin.
    """
    setup_logging()

    if len(sys.argv) >= 2 and sys.argv[1] != "-":
        raw = sys.argv[1]
    elif not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
    else:
        log.error("No trigger payload provided. Pass a JSON string as argv[1] or pipe via stdin.")
        sys.exit(1)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        log.error("Invalid JSON payload: %s", e)
        sys.exit(1)

    config_path = Path(payload.get("config", str(DEFAULT_CONFIG)))
    setup_logging(config_path)  # re-apply in case config path came from payload

    try:
        run_config = resolve_inputs(
            rfp_path=payload.get("rfp_path"),
            client_name=payload.get("client_name"),
            rfp_tone=payload.get("rfp_tone"),
            output_dir=payload.get("output_dir"),
            config_path=config_path,
        )
    except (ValueError, FileNotFoundError) as e:
        log.error("Bad trigger payload: %s", e)
        sys.exit(1)

    inputs = build_crew_inputs(run_config)
    inputs["crewai_trigger_payload"] = payload

    log.info("Trigger: starting crew — client: %s | RFP: %s", run_config.client_name, run_config.rfp_path.name)
    try:
        result = Draftr().crew().kickoff(inputs=inputs)
        log.info("Trigger crew finished. Outputs in: %s", run_config.output_dir)
        return result
    except Exception as e:
        log.error("Trigger crew failed: %s", e, exc_info=True)
        raise
