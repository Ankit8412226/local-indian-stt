"""Main transcription CLI.

Usage:
    python -m src.transcribe audio.wav
    python -m src.transcribe audio.wav --model indic-conformer
    python -m src.transcribe audio.wav --model whisper-medium --language hi
    python -m src.transcribe audio.wav --model sravaani --no-db
    python -m src.transcribe audio.wav --device cpu

The transcription engine is independent of the CLI and database layer.
MongoDB storage is optional via --no-db.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Load .env before any src imports
_root = Path(__file__).resolve().parent.parent
_env = _root / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.utils.logger import get_logger, log_inference_event

console = Console()
_logger = get_logger(__name__)


@click.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option(
    "--model", "model_id",
    default=None,
    help="Model ID (indic-conformer|sravaani|whisper-medium|whisper-small|"
         "whisper-large-v3-turbo|indic-whisper-hindi|indic-whisper-medium-hindi). "
         "Default: whisper-medium",
)
@click.option("--language", default="hi", show_default=True, help="BCP-47 language code")
@click.option("--device", default="auto", show_default=True, help="Device: auto|cpu|cuda|mps")
@click.option(
    "--output-dir",
    default="outputs",
    show_default=True,
    help="Directory for transcript output files",
)
@click.option("--no-db", "no_db", is_flag=True, help="Skip MongoDB storage")
@click.option("--no-normalize", "no_normalize", is_flag=True, help="Skip audio normalization")
def transcribe(
    audio_path: str,
    model_id: str | None,
    language: str,
    device: str,
    output_dir: str,
    no_db: bool,
    no_normalize: bool,
) -> None:
    """Transcribe AUDIO_PATH using a local Indian ASR model.

    Outputs transcript to stdout and saves to OUTPUT_DIR/<filename>.txt.
    """
    import time

    request_id = str(uuid.uuid4())
    audio_file = Path(audio_path)

    console.rule(f"[bold cyan]Local Indian STT[/bold cyan]")
    console.print(f"  [dim]Request ID:[/dim] {request_id}")
    console.print(f"  [dim]Audio file:[/dim] {audio_file.name}")
    console.print(f"  [dim]Language:  [/dim] {language}")
    console.print(f"  [dim]Device:    [/dim] {device}")

    # ── Step 1: Audio processing ─────────────────────────────────────────
    console.print("\n[cyan]→ Processing audio...[/cyan]")
    t_pre = time.perf_counter()
    try:
        from src.audio.processor import process_audio
        processed = process_audio(str(audio_file), normalize=not no_normalize)
    except Exception as e:
        console.print(f"[bold red]✗ Audio processing failed: {e}[/bold red]")
        sys.exit(1)
    preprocessing_time = time.perf_counter() - t_pre

    meta = processed.metadata
    quality = processed.quality

    # ── Step 2: Load model ───────────────────────────────────────────────
    console.print("[cyan]→ Loading model...[/cyan]")
    t_load = time.perf_counter()
    try:
        from src.models.factory import get_asr_model
        model = get_asr_model(model_id, device=None if device == "auto" else None)
        from src.utils.device import resolve_device
        torch_device = resolve_device(device)
        model = get_asr_model(model_id, device=torch_device)
        model.load_model()
    except Exception as e:
        console.print(f"[bold red]✗ Model loading failed: {e}[/bold red]")
        sys.exit(1)
    model_load_time = time.perf_counter() - t_load
    console.print(f"  [dim]Model loaded in {model_load_time:.2f}s[/dim]")

    # ── Step 3: Transcription ────────────────────────────────────────────
    console.print("[cyan]→ Transcribing...[/cyan]")
    try:
        output = model.transcribe(
            waveform=processed.waveform,
            sample_rate=processed.sample_rate,
            request_id=request_id,
            language=language,
            audio_duration=meta.duration,
            audio_channels=meta.channels,
            is_telephone=processed.is_telephone,
            preprocessing_time=preprocessing_time,
        )
    except Exception as e:
        console.print(f"[bold red]✗ Inference failed: {e}[/bold red]")
        log_inference_event(
            _logger,
            request_id=request_id,
            model_id=model_id or "unknown",
            audio_path=str(audio_file),
            duration=meta.duration,
            preprocessing_time=preprocessing_time,
            inference_time=None,
            rtf=None,
            device=device,
            success=False,
            error=str(e),
        )
        sys.exit(1)

    # ── Step 4: Display results ──────────────────────────────────────────
    _display_results(output, meta, quality, model_load_time)

    # ── Step 5: Save transcript file ─────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{audio_file.stem}.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Request ID: {request_id}\n")
        f.write(f"Model:      {output.model_id}\n")
        f.write(f"Language:   {output.language}\n")
        f.write(f"Duration:   {output.audio_duration:.2f}s\n")
        f.write(f"RTF:        {output.rtf:.4f}\n")
        f.write(f"Timestamp:  {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"\n{'='*60}\n\n")
        f.write(output.transcript)
        f.write("\n")

    console.print(f"\n[green]✓ Transcript saved:[/green] {out_file}")

    # ── Step 6: MongoDB storage (optional) ───────────────────────────────
    if not no_db:
        _save_to_db(output, audio_path=str(audio_file), request_id=request_id)

    log_inference_event(
        _logger,
        request_id=request_id,
        model_id=output.model_id,
        audio_path=str(audio_file),
        duration=output.audio_duration,
        preprocessing_time=output.preprocessing_time,
        inference_time=output.inference_time,
        rtf=output.rtf,
        device=output.device,
        success=True,
        transcript=output.transcript,
    )


def _display_results(output, meta, quality, model_load_time: float) -> None:
    """Print a rich summary table and transcript panel."""
    # Audio metadata table
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    tbl.add_column(style="dim", no_wrap=True)
    tbl.add_column(style="bold")
    tbl.add_row("Audio duration", f"{output.audio_duration:.2f}s")
    tbl.add_row("Original sample rate", f"{meta.sample_rate} Hz")
    tbl.add_row("Channels", str(meta.channels))
    tbl.add_row("Telephone audio", "Yes ☎" if meta.is_telephone else "No")
    tbl.add_row("Quality flag", quality.quality_flag)
    tbl.add_row("Model", output.model_id)
    tbl.add_row("Device", output.device)
    tbl.add_row("Model load time", f"{model_load_time:.2f}s")
    tbl.add_row("Preprocessing time", f"{output.preprocessing_time:.3f}s")
    tbl.add_row("Inference time", f"{output.inference_time:.3f}s")
    tbl.add_row("Total time", f"{output.total_time:.3f}s")
    tbl.add_row(
        "Real-Time Factor (RTF)",
        f"{output.rtf:.4f}  [dim]({'faster' if output.rtf < 1 else 'slower'} than real-time)[/dim]",
    )
    console.print(tbl)

    console.print(
        Panel(
            output.transcript or "[dim](empty transcript)[/dim]",
            title="[bold green]Transcript[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


def _save_to_db(output, audio_path: str, request_id: str) -> None:
    """Save job and result to MongoDB. Non-fatal on failure."""
    from datetime import datetime, timezone
    try:
        from src.database.connection import get_db, ping_db
        if not ping_db():
            _logger.warning(
                "MongoDB unavailable — transcription result not stored. "
                "Use --no-db to suppress this warning."
            )
            return
        db = get_db()
        from src.database.repository import TranscriptionRepository
        repo = TranscriptionRepository(db)
        now = datetime.now(timezone.utc)

        repo.create_job({
            "request_id": request_id,
            "audio_path": audio_path,
            "model_id": output.model_id,
            "model_version": output.model_version,
            "language": output.language,
            "status": "completed",
            "created_at": now,
            "completed_at": now,
        })

        repo.save_result({
            "request_id": request_id,
            "model_id": output.model_id,
            "model_version": output.model_version,
            "transcript": output.transcript,
            "audio": {
                "duration": output.audio_duration,
                "sample_rate": output.audio_sample_rate,
                "channels": output.audio_channels,
                "is_telephone": output.is_telephone,
            },
            "processing": {
                "preprocessing_time": output.preprocessing_time,
                "inference_time": output.inference_time,
                "total_time": output.total_time,
                "rtf": output.rtf,
            },
            "device": output.device,
        })
        console.print("[dim]  ✓ Stored in MongoDB[/dim]")
    except Exception as e:
        _logger.warning("MongoDB write failed (result still returned): %s", e)
        console.print(f"[yellow]  ⚠ MongoDB write failed: {e}[/yellow]")


if __name__ == "__main__":
    transcribe()
