/**
 * What these pin: the colour rules from the brief, as properties rather than as
 * a swatch somebody eyeballed.
 *
 * The one worth stating aloud is the collision rule. "Every agent gets a
 * different colour" is only true up to eight, and a test that asserted it
 * unconditionally would be asserting something false. What is asserted instead
 * is: distinct up to the size of the palette, deterministic always, and never
 * the sole carrier of meaning — which is why the legend and the inspector
 * exist.
 */
import { describe, expect, it } from 'vitest';
import {
  AGENT_PALETTE,
  CANONICAL_COLOUR,
  agentHash,
  assignAgentColours,
  colourOf,
  fade,
} from './agentColor';

describe('the palette', () => {
  it('is exactly the eight the brief names, in order', () => {
    expect([...AGENT_PALETTE]).toEqual([
      '#60A5FA',
      '#2DD4BF',
      '#A78BFA',
      '#F472B6',
      '#FBBF24',
      '#FB923C',
      '#4ADE80',
      '#F87171',
    ]);
  });

  it('keeps the shared canonical slate out of the agent palette', () => {
    expect(CANONICAL_COLOUR).toBe('#94A3B8');
    expect([...AGENT_PALETTE]).not.toContain(CANONICAL_COLOUR);
  });
});

describe('assignment', () => {
  it('gives eight agents eight different colours', () => {
    const agents = ['ag-a', 'ag-b', 'ag-c', 'ag-d', 'ag-e', 'ag-f', 'ag-g', 'ag-h'];
    const colours = assignAgentColours(agents);
    expect(new Set(colours.values()).size).toBe(8);
  });

  it('gives a handful of agents different colours, which a bare hash would not', () => {
    // The case that motivated the probe: four agents into eight slots collide
    // about forty per cent of the time under a plain modulo.
    for (const agents of [
      ['planner', 'researcher', 'writer', 'checker'],
      ['ag-1', 'ag-2', 'ag-3'],
      ['alpha', 'beta'],
    ]) {
      const colours = assignAgentColours(agents);
      expect(new Set(colours.values()).size).toBe(agents.length);
    }
  });

  it('is deterministic — the same set gives the same answer every time', () => {
    const agents = ['ag-c', 'ag-a', 'ag-b'];
    const first = assignAgentColours(agents);
    const second = assignAgentColours([...agents].reverse());
    expect([...second.entries()].sort()).toEqual([...first.entries()].sort());
  });

  it('does not depend on the order rows arrived in', () => {
    const sorted = assignAgentColours(['a', 'b', 'c']);
    const shuffled = assignAgentColours(['c', 'a', 'b']);
    expect(shuffled.get('b')).toBe(sorted.get('b'));
  });

  it('reuses a colour past eight agents rather than inventing a ninth', () => {
    const agents = Array.from({ length: 12 }, (_, i) => `ag-${i}`);
    const colours = assignAgentColours(agents);
    expect(colours.size).toBe(12);
    // Pigeonhole: twelve into eight must repeat. What matters is that every
    // colour is still one of the eight.
    for (const colour of colours.values()) {
      expect([...AGENT_PALETTE]).toContain(colour);
    }
    expect(new Set(colours.values()).size).toBeLessThanOrEqual(8);
  });

  it('hashes stably, so a colour survives a restart', () => {
    // A hash swapped for insertion order or Math.random would pass every other
    // test in this file and fail this one.
    expect(agentHash('ag-a')).toBe(agentHash('ag-a'));
    expect(agentHash('ag-a')).not.toBe(agentHash('ag-b'));
    expect(Number.isInteger(agentHash('planner'))).toBe(true);
  });
});

describe('canonical material belongs to nobody', () => {
  it('draws in the shared slate, not in whoever cited it first', () => {
    const colours = assignAgentColours(['ag-a']);
    expect(colourOf(colours, null)).toBe(CANONICAL_COLOUR);
  });

  it('falls back to the slate for an agent that is not in the legend', () => {
    const colours = assignAgentColours(['ag-a']);
    expect(colourOf(colours, 'a-stranger')).toBe(CANONICAL_COLOUR);
  });

  it('gives a known agent its own colour', () => {
    const colours = assignAgentColours(['ag-a', 'ag-b']);
    expect(colourOf(colours, 'ag-a')).toBe(colours.get('ag-a'));
    expect([...AGENT_PALETTE]).toContain(colourOf(colours, 'ag-a'));
  });
});

describe('fading', () => {
  it('appends an alpha byte the canvas understands', () => {
    expect(fade('#60A5FA', 1)).toBe('#60A5FAff');
    expect(fade('#60A5FA', 0)).toBe('#60A5FA00');
    expect(fade('#60A5FA', 0.5)).toBe('#60A5FA80');
  });

  it('clamps rather than emitting a malformed colour', () => {
    expect(fade('#60A5FA', 5)).toBe('#60A5FAff');
    expect(fade('#60A5FA', -2)).toBe('#60A5FA00');
    // Always eight hex digits after the hash, whatever it was given.
    expect(fade('#60A5FA', 0.04)).toHaveLength(9);
  });
});
