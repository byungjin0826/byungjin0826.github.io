/* 신병진 Design System 프리미티브 — Tailwind 이식판.
   훅을 쓰지 않아 서버/클라이언트 트리 어디서든 사용 가능. */
import { ReactNode } from 'react'
import Link from '@/components/Link'

function cn(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(' ')
}

/* ---------- Eyebrow (mono 라벨) ---------- */
export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('ds-eyebrow', className)}>{children}</div>
}

/* ---------- Button ---------- */
type ButtonVariant = 'primary' | 'solid' | 'secondary' | 'ghost'
type ButtonSize = 'sm' | 'md' | 'lg'

const BTN_BASE =
  'inline-flex items-center justify-center gap-2 rounded-md font-semibold leading-none tracking-[-0.01em] transition-colors'
const BTN_SIZE: Record<ButtonSize, string> = {
  sm: 'px-3.5 py-[7px] text-[0.8125rem]',
  md: 'px-5 py-2.5 text-[0.9375rem]',
  lg: 'px-[26px] py-[13px] text-base',
}
const BTN_VARIANT: Record<ButtonVariant, string> = {
  primary: 'border-[1.5px] border-accent bg-accent text-on-accent hover:border-accent-hover hover:bg-accent-hover',
  solid: 'border-[1.5px] border-ink bg-ink text-ink-inverse hover:opacity-90',
  secondary: 'border-[1.5px] border-line-strong bg-transparent text-ink hover:border-ink hover:bg-surface-2',
  ghost: 'border-[1.5px] border-transparent bg-transparent text-ink hover:bg-surface-2',
}

interface ButtonProps {
  children: ReactNode
  variant?: ButtonVariant
  size?: ButtonSize
  full?: boolean
  href?: string
  className?: string
  type?: 'button' | 'submit' | 'reset'
  target?: string
  rel?: string
  disabled?: boolean
  onClick?: React.MouseEventHandler
}

export function Button({
  children,
  variant = 'primary',
  size = 'md',
  full = false,
  href,
  className,
  type = 'button',
  target,
  rel,
  disabled,
  onClick,
}: ButtonProps) {
  const cls = cn(BTN_BASE, BTN_SIZE[size], BTN_VARIANT[variant], full && 'w-full', className)
  if (href) {
    return (
      <Link href={href} className={cls} target={target} rel={rel} onClick={onClick}>
        {children}
      </Link>
    )
  }
  return (
    <button className={cls} type={type} disabled={disabled} onClick={onClick}>
      {children}
    </button>
  )
}

/* ---------- IconButton ---------- */
type IconButtonVariant = 'ghost' | 'outline' | 'solid' | 'accent'
const ICONBTN_VARIANT: Record<IconButtonVariant, string> = {
  ghost: 'border-[1.5px] border-transparent text-ink hover:bg-surface-2',
  outline: 'border-[1.5px] border-line text-ink hover:border-line-strong hover:bg-surface-2',
  solid: 'border-[1.5px] border-ink bg-ink text-ink-inverse hover:opacity-90',
  accent: 'border-[1.5px] border-accent bg-accent text-on-accent hover:bg-accent-hover',
}

export function IconButton({
  children,
  variant = 'ghost',
  label,
  className,
  ...rest
}: {
  children: ReactNode
  variant?: IconButtonVariant
  label: string
  className?: string
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      aria-label={label}
      title={label}
      className={cn(
        'inline-flex h-[38px] w-[38px] items-center justify-center rounded-md transition-colors',
        ICONBTN_VARIANT[variant],
        className
      )}
      {...rest}
    >
      {children}
    </button>
  )
}

/* ---------- Chip (카테고리/토픽 태그) ---------- */
type ChipVariant = 'solid' | 'outline'
const CHIP_BASE =
  'inline-flex items-center gap-1.5 rounded-sm px-3 py-[5px] text-[0.8125rem] font-semibold leading-none tracking-[-0.01em] whitespace-nowrap transition-colors'

function chipClasses(variant: ChipVariant, active: boolean, interactive: boolean) {
  if (active) return 'border-[1.5px] border-ink bg-ink text-ink-inverse'
  const base =
    variant === 'solid'
      ? 'border-[1.5px] border-transparent bg-surface-2 text-ink'
      : 'border-[1.5px] border-line-strong bg-transparent text-ink'
  const hover = interactive
    ? variant === 'solid'
      ? 'hover:bg-accent-soft hover:text-accent-text'
      : 'hover:border-accent-press hover:bg-accent-soft hover:text-accent-text'
    : ''
  return cn(base, hover)
}

export function Chip({
  children,
  variant = 'outline',
  active = false,
  href,
  onClick,
  className,
}: {
  children: ReactNode
  variant?: ChipVariant
  active?: boolean
  href?: string
  onClick?: () => void
  className?: string
}) {
  const interactive = Boolean(href || onClick)
  const cls = cn(CHIP_BASE, chipClasses(variant, active, interactive), interactive && 'cursor-pointer', className)
  if (href) {
    return (
      <Link href={href} className={cls}>
        {children}
      </Link>
    )
  }
  if (onClick) {
    return (
      <button type="button" onClick={onClick} className={cls}>
        {children}
      </button>
    )
  }
  return <span className={cls}>{children}</span>
}

/* ---------- Avatar (이니셜 or 이미지) ---------- */
const AV_SIZE: Record<string, string> = {
  xs: 'h-6 w-6 text-[10px]',
  sm: 'h-8 w-8 text-[13px]',
  md: 'h-10 w-10 text-base',
  lg: 'h-14 w-14 text-[22px]',
  xl: 'h-20 w-20 text-[32px]',
}

export function Avatar({
  name = '',
  src,
  size = 'md',
}: {
  name?: string
  src?: string
  size?: keyof typeof AV_SIZE
}) {
  const initials =
    name
      .trim()
      .split(/\s+/)
      .map((w) => w[0])
      .slice(0, 2)
      .join('')
      .toUpperCase() || '·'
  return (
    <span
      className={cn(
        'inline-flex flex-shrink-0 select-none items-center justify-center overflow-hidden rounded-full bg-accent font-display font-bold tracking-[-0.02em] text-on-accent',
        AV_SIZE[size]
      )}
    >
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src} alt={name} className="h-full w-full object-cover" />
      ) : (
        initials
      )}
    </span>
  )
}

/* ---------- Divider (라벨 삽입 가능) ---------- */
export function Divider({
  label,
  variant = 'hair',
  className,
}: {
  label?: ReactNode
  variant?: 'hair' | 'rule'
  className?: string
}) {
  const line = variant === 'rule' ? 'border-rule' : 'border-line'
  const weight = variant === 'rule' ? 'border-t-2' : 'border-t'
  if (label) {
    return (
      <div className={cn('flex items-center gap-4', className)}>
        <span className={cn('h-0 flex-1', weight, line)} />
        <span className="ds-eyebrow whitespace-nowrap">{label}</span>
        <span className={cn('h-0 flex-1', weight, line)} />
      </div>
    )
  }
  return <hr className={cn(weight, line, 'm-0', className)} />
}

/* ---------- Callout (본문 강조 · MDX용) ---------- */
const CALLOUT_TONE: Record<string, string> = {
  accent: 'border-accent bg-accent-soft',
  info: 'border-info bg-surface-2',
  warning: 'border-warning bg-surface-2',
  neutral: 'border-line-strong bg-surface-2',
}

export function Callout({
  children,
  tone = 'accent',
  title,
}: {
  children: ReactNode
  tone?: keyof typeof CALLOUT_TONE
  title?: string
}) {
  return (
    <div className={cn('my-6 flex gap-3 rounded-sm border-l-[3px] px-5 py-4', CALLOUT_TONE[tone])}>
      <div>
        {title && <div className="mb-1 font-bold text-ink">{title}</div>}
        <div className="text-[0.9375rem] leading-relaxed text-ink">{children}</div>
      </div>
    </div>
  )
}
